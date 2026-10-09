from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def policy_token_stats(model, seq, attn, pw, resp_ids, with_entropy=False):
    """log pi(a_t | s_t) for every response token, plus full-distribution entropy if requested."""
    pos = (attn.cumsum(-1) - 1).clamp_min(0)          # correct positions under left padding
    logits = model(input_ids=seq, attention_mask=attn, position_ids=pos, use_cache=False).logits
    logits = logits[:, pw - 1: pw - 1 + resp_ids.shape[1], :].float()
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1)
    ent = -(logp_all.exp() * logp_all).sum(-1) if with_entropy else None
    return logp, ent


def response_values(value_model, seq, attn, pw, n):
    """V(s_t) for each response step: hidden state at the position that predicts token t."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        v = token_values(value_model, seq, attn)
    return v[:, pw - 1: pw - 1 + n].float()


def explained_variance(values, returns, mask):
    m = mask.bool()
    v, r = values[m], returns[m]
    var_r = r.var(unbiased=False)
    return float(1.0 - (r - v).var(unbiased=False) / var_r) if var_r > 1e-8 else float("nan")


@torch.no_grad()
def collect_rollout(policy, value_model, reward_model, reward_tokenizer, tokenizer, msgs, cfg, max_new):
    g = cfg["generation"]
    gen = batch_generate(policy, tokenizer, msgs, int(cfg["max_prompt_length"]), max_new,
                         temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]))
    seq, attn, pw, resp = gen["sequences"].clone(), gen["attention_mask"].clone(), gen["prompt_width"], gen["response_ids"].clone()  # leave inference_mode tensors behind
    mask = gen["response_mask"]
    old_logp, ent = policy_token_stats(policy, seq, attn, pw, resp, with_entropy=True)
    with reference_mode(policy):
        ref_logp, _ = policy_token_stats(policy, seq, attn, pw, resp)
    policy.eval()
    values = response_values(value_model, seq, attn, pw, resp.shape[1])
    raw = score_reward_pairs(reward_model, reward_tokenizer, msgs, gen["responses"],
                             max_length=int(cfg.get("reward_max_length", 1024))).to(old_logp.device)
    pen = float(cfg.get("missing_eos_penalty", 0.0))
    eff = raw - torch.tensor([0.0 if t else pen for t in gen["terminated_with_eos"]], device=raw.device)
    return {"seq": seq, "attn": attn, "pw": pw, "resp": resp, "mask": mask, "old_logp": old_logp,
            "ref_logp": ref_logp, "entropy": ent, "values": values, "reward_raw": raw, "reward_eff": eff,
            "responses": gen["responses"], "truncated": gen["truncated"], "terminated": gen["terminated_with_eos"]}


def ppo_update(policy, value_model, pol_opt, val_opt, ro, adv, returns, cfg, eps):
    mask = ro["mask"]
    max_norm, vcoef = float(cfg["max_grad_norm"]), float(cfg["value_coef"])
    pol_params, val_params = trainable_parameters(policy), trainable_parameters(value_model)
    stats = []
    for epoch in range(int(cfg["ppo_epochs"])):
        new_logp, _ = policy_token_stats(policy, ro["seq"], ro["attn"], ro["pw"], ro["resp"])
        pol_loss, ratio, clip_frac = ppo_policy_loss(new_logp, ro["old_logp"], adv, mask, eps)
        pol_opt.zero_grad(set_to_none=True)
        pol_loss.backward()
        pg = torch.nn.utils.clip_grad_norm_(pol_params, max_norm)
        pol_ok = bool(torch.isfinite(pg))
        if pol_ok:
            pol_opt.step()
        pol_opt.zero_grad(set_to_none=True)

        new_v = response_values(value_model, ro["seq"], ro["attn"], ro["pw"], ro["resp"].shape[1])
        v_loss = value_mse_loss(new_v, returns, mask)
        val_opt.zero_grad(set_to_none=True)
        (vcoef * v_loss).backward()
        vg = torch.nn.utils.clip_grad_norm_(val_params, max_norm)
        val_ok = bool(torch.isfinite(vg))
        if val_ok:
            val_opt.step()
        val_opt.zero_grad(set_to_none=True)

        r = ratio[mask.bool()]
        stats.append({
            "epoch": epoch + 1, "policy_loss": pol_loss.item(), "clip_fraction": clip_frac.item(),
            "ratio_mean": r.mean().item(), "ratio_min": r.min().item(), "ratio_max": r.max().item(),
            "approx_kl_old": masked_mean(ro["old_logp"] - new_logp.detach(), mask).item(),
            "value_loss": v_loss.item(), "policy_grad_norm": float(pg), "value_grad_norm": float(vg),
            "policy_step_ok": pol_ok, "value_step_ok": val_ok,
        })
        del new_logp, new_v
    return stats


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    policy.eval()                      # dropout off: ratio == 1 exactly before any update
    value_model = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"],
                                   train_mode=cfg.get("value_train_mode", "head_only"))
    # Trainable critic head in fp32 (fp16 Adam states underflow); critic forwards run under autocast.
    for _, p in value_model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    value_model.eval()
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
    value_optimizer = AdamW(
        value_parameter_groups(value_model, lora_lr=float(cfg["value_lora_learning_rate"]),
                               head_lr=float(cfg["value_head_learning_rate"])),
        weight_decay=0.0,
    )
    return {"cfg": cfg, "tokenizer": tokenizer, "policy": policy, "value_model": value_model,
            "reward_model": reward_model, "reward_tokenizer": reward_tokenizer, "prompt_rows": prompts,
            "policy_optimizer": policy_optimizer, "value_optimizer": value_optimizer}


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    n_updates, eps, beta_kl = int(cfg["updates"]), float(cfg["clip_epsilon"]), float(cfg["kl_beta"])

    default_out = cfg["output"] if run_name == "standard" else f"outputs/task2_ppo/{run_name}"
    out = repo_path(output or default_out)
    out.mkdir(parents=True, exist_ok=True)
    res = repo_path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    log_path, roll_path = res / f"{run_name}_train_log.jsonl", res / f"{run_name}_rollouts.jsonl"
    for p in (log_path, roll_path):
        if p.exists():
            p.unlink()

    policy, value_model = bundle["policy"], bundle["value_model"]
    pol_opt, val_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    rm, rm_tok, tok, rows = bundle["reward_model"], bundle["reward_tokenizer"], bundle["tokenizer"], bundle["prompt_rows"]
    seed, per_update = int(cfg["seed"]), int(cfg["prompts_per_update"])
    max_new = int(cfg["max_response_length"])
    order = np.random.default_rng(seed).permutation(len(rows))   # identical prompt order for every fork

    save_json(res / f"{run_name}_config.json", {
        "run_name": run_name, "updates": n_updates, "clip_epsilon": eps, "kl_beta": beta_kl,
        "prompts_per_update": per_update, "ppo_epochs": int(cfg["ppo_epochs"]),
        "policy_learning_rate": float(cfg["policy_learning_rate"]),
        "value_lora_learning_rate": float(cfg["value_lora_learning_rate"]),
        "value_head_learning_rate": float(cfg["value_head_learning_rate"]), "value_train_mode": cfg.get("value_train_mode"),
        "gamma": float(cfg["gamma"]), "gae_lambda": float(cfg["gae_lambda"]), "value_coef": float(cfg["value_coef"]),
        "missing_eos_penalty": float(cfg.get("missing_eos_penalty", 0.0)), "max_response_length": max_new,
        "max_prompt_length": int(cfg["max_prompt_length"]), "max_grad_norm": float(cfg["max_grad_norm"]),
        "generation": cfg["generation"], "seed": seed, "start_policy": cfg["paths"]["ppo_midpoint_policy"],
        "start_value": cfg["paths"]["ppo_midpoint_value"], "output": str(out),
    })

    timer = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for u in range(n_updates):
        idx = [int(i) for i in order[u * per_update:(u + 1) * per_update]]
        msgs = [prompt_messages(rows[i]) for i in idx]
        set_seed(seed + u)                       # same sampling randomness per update across forks
        ro = collect_rollout(policy, value_model, rm, rm_tok, tok, msgs, cfg, max_new)
        mask = ro["mask"]
        rewards = shaped_rewards(ro["reward_eff"], ro["old_logp"], ro["ref_logp"], mask, beta_kl)
        adv_raw, returns = compute_gae(rewards, ro["values"], mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
        adv = normalize_advantages(adv_raw, mask)
        ep = ppo_update(policy, value_model, pol_opt, val_opt, ro, adv, returns, cfg, eps)

        kl_tok = ro["old_logp"] - ro["ref_logp"]
        rec = {
            "update": u + 1, "prompt_indices": idx,
            "reward_raw": ro["reward_raw"].mean().item(), "reward_effective": ro["reward_eff"].mean().item(),
            "kl_token_mean": masked_mean(kl_tok, mask).item(), "kl_seq_sum": (kl_tok * mask).sum(-1).mean().item(),
            "entropy": masked_mean(ro["entropy"], mask).item(),
            "response_length": float(mask.sum(-1).float().mean().item()),
            "truncated_rate": float(np.mean(ro["truncated"])), "eos_rate": float(np.mean(ro["terminated"])),
            "value_mean": masked_mean(ro["values"], mask).item(), "return_mean": masked_mean(returns, mask).item(),
            "value_explained_variance": explained_variance(ro["values"], returns, mask),
            "adv_raw_mean": masked_mean(adv_raw, mask).item(),
            "policy_loss": float(np.mean([e["policy_loss"] for e in ep])),
            "value_loss": float(np.mean([e["value_loss"] for e in ep])),
            "clip_fraction": float(np.mean([e["clip_fraction"] for e in ep])),
            "clip_fraction_last_epoch": ep[-1]["clip_fraction"],
            "ratio_min": min(e["ratio_min"] for e in ep), "ratio_max": max(e["ratio_max"] for e in ep),
            "approx_kl_old_last_epoch": ep[-1]["approx_kl_old"],
            "policy_grad_norm": float(np.mean([e["policy_grad_norm"] for e in ep])),
            "value_grad_norm": float(np.mean([e["value_grad_norm"] for e in ep])),
            "skipped_steps": sum((not e["policy_step_ok"]) + (not e["value_step_ok"]) for e in ep),
            "epochs": ep, "elapsed_sec": timer(),
            "peak_vram_gib_so_far": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        }
        append_jsonl(log_path, rec)
        for j, i in enumerate(idx):
            append_jsonl(roll_path, {"update": u + 1, "prompt_index": i, "prompt_id": rows[i].get("prompt_id"),
                                     "response": ro["responses"][j], "reward_raw": ro["reward_raw"][j].item(),
                                     "reward_effective": ro["reward_eff"][j].item(),
                                     "length": int(mask[j].sum().item()), "truncated": ro["truncated"][j]})
        print(f"update {u + 1:2d}/{n_updates} | reward {rec['reward_effective']:+.3f} | KL/tok {rec['kl_token_mean']:+.4f} | "
              f"clip {rec['clip_fraction_last_epoch']:.3f} | vloss {rec['value_loss']:.3f} | EV {rec['value_explained_variance']:+.2f} | "
              f"len {rec['response_length']:.0f} | {rec['elapsed_sec'] / 60:.1f} min", flush=True)
        del ro, rewards, adv_raw, returns, adv

    policy.save_pretrained(str(out))
    value_model.save_pretrained(str(out / "value_critic"))
    summary = {"run_name": run_name, "updates": n_updates, "clip_epsilon": eps, "kl_beta": beta_kl,
               "wall_clock_sec": timer(),
               "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
               "policy_adapter": str(out), "value_adapter": str(out / "value_critic")}
    save_json(res / f"{run_name}_train_summary.json", summary)
    print(summary, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
