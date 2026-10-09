from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

UNINFORMATIVE_TOL = 1e-6   # a group is uninformative when its reward std <= this tolerance


def train_without_dropout(model):
    """Train mode (keeps gradient checkpointing on) with every dropout disabled, so the
    importance ratio is exactly 1 before the update instead of being inflated by dropout noise."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()


def policy_token_stats(model, seq, attn, pw, resp_ids, with_entropy=False):
    pos = (attn.cumsum(-1) - 1).clamp_min(0)
    logits = model(input_ids=seq, attention_mask=attn, position_ids=pos, use_cache=False).logits
    logits = logits[:, pw - 1: pw - 1 + resp_ids.shape[1], :].float()
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1)
    ent = -(logp_all.exp() * logp_all).sum(-1) if with_entropy else None
    return logp, ent


@torch.no_grad()
def collect_group_rollout(policy, rm, rm_tok, tok, prompt_msgs, cfg, k):
    msgs = [m for m in prompt_msgs for _ in range(k)]
    group_ids = torch.tensor([g for g in range(len(prompt_msgs)) for _ in range(k)])
    g = cfg["generation"]
    gen = batch_generate(policy, tok, msgs, int(cfg["max_prompt_length"]), int(cfg["max_completion_length"]),
                         temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]))
    train_without_dropout(policy)
    seq, attn, pw, resp = gen["sequences"].clone(), gen["attention_mask"].clone(), gen["prompt_width"], gen["response_ids"].clone()  # leave inference_mode tensors behind
    old_logp, ent = policy_token_stats(policy, seq, attn, pw, resp, with_entropy=True)
    with reference_mode(policy):
        ref_logp, _ = policy_token_stats(policy, seq, attn, pw, resp)
    train_without_dropout(policy)
    rewards = score_reward_pairs(rm, rm_tok, msgs, gen["responses"],
                                 max_length=int(cfg.get("reward_max_length", 1024))).float().to(old_logp.device)
    return {"seq": seq, "attn": attn, "pw": pw, "resp": resp, "mask": gen["response_mask"],
            "old_logp": old_logp, "ref_logp": ref_logp, "entropy": ent, "rewards": rewards,
            "group_ids": group_ids.to(old_logp.device), "responses": gen["responses"],
            "truncated": gen["truncated"], "lengths": gen["response_lengths"]}


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    train_without_dropout(policy)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {"cfg": cfg, "tokenizer": tokenizer, "policy": policy, "reward_model": reward_model,
            "reward_tokenizer": reward_tokenizer, "prompt_rows": prompts, "optimizer": optimizer}


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    n_updates, eps, beta = int(cfg["updates"]), float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    k, per_update, L = int(cfg["num_generations"]), int(cfg["prompts_per_update"]), int(cfg["max_completion_length"])
    mask_trunc = bool(cfg.get("mask_truncated_completions", False))

    default_out = cfg["output"] if run_name == "standard" else f"outputs/task3_grpo/{run_name}"
    out = repo_path(output or default_out)
    out.mkdir(parents=True, exist_ok=True)
    res = repo_path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    log_path, comp_path = res / f"{run_name}_train_log.jsonl", res / f"{run_name}_completions.jsonl"
    for p in (log_path, comp_path):
        if p.exists():
            p.unlink()

    policy, opt = bundle["policy"], bundle["optimizer"]
    rm, rm_tok, tok, rows = bundle["reward_model"], bundle["reward_tokenizer"], bundle["tokenizer"], bundle["prompt_rows"]
    seed = int(cfg["seed"])
    order = np.random.default_rng(seed).permutation(len(rows))      # identical prompt order for every fork
    params = trainable_parameters(policy)

    save_json(res / f"{run_name}_config.json", {
        "run_name": run_name, "loss_type": loss_type, "updates": n_updates, "num_generations": k,
        "prompts_per_update": per_update, "policy_epochs": int(cfg["policy_epochs"]),
        "learning_rate": float(cfg["learning_rate"]), "clip_epsilon": eps, "kl_beta": beta,
        "max_prompt_length": int(cfg["max_prompt_length"]), "max_completion_length": L,
        "mask_truncated_completions": mask_trunc, "max_grad_norm": float(cfg["max_grad_norm"]),
        "uninformative_tolerance": UNINFORMATIVE_TOL, "generation": cfg["generation"], "seed": seed,
        "start_policy": cfg["paths"]["grpo_midpoint_policy"], "output": str(out),
    })

    timer = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for u in range(n_updates):
        idx = [int(i) for i in order[u * per_update:(u + 1) * per_update]]
        set_seed(seed + u)                      # same sampling randomness per update across forks
        ro = collect_group_rollout(policy, rm, rm_tok, tok, [prompt_messages(rows[i]) for i in idx], cfg, k)
        n = ro["rewards"].shape[0]
        token_mask = mask_truncated_sequences(ro["mask"], ro["truncated"]) if mask_trunc else ro["mask"]
        adv = group_relative_advantages(ro["rewards"], ro["group_ids"])

        diag, gn, stepped = None, float("nan"), False
        for _ in range(int(cfg["policy_epochs"])):
            new_logp, _ = policy_token_stats(policy, ro["seq"], ro["attn"], ro["pw"], ro["resp"])
            loss, diag = grpo_policy_loss(new_logp, ro["old_logp"], adv, token_mask, ro["ref_logp"], eps, beta,
                                          loss_type=loss_type, max_completion_length=L)
            opt.zero_grad(set_to_none=True)
            if token_mask.sum() > 0:
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
                if np.isfinite(gn):
                    opt.step()
                    stepped = True
            opt.zero_grad(set_to_none=True)
            del new_logp

        # ---- group statistics ----
        stds = [ro["rewards"][ro["group_ids"] == g].std(unbiased=False).item() for g in torch.unique(ro["group_ids"])]
        # ---- length-conditioned gradient allocation (ratio ~ 1, so per-token gradient scale ~ |A| * weight) ----
        T = token_mask.sum(-1)
        comp_recs = []
        for j in range(n):
            t, a = int(T[j].item()), float(adv[j].item())
            tok_w = 0.0 if t == 0 else ((1.0 / t) if loss_type == "grpo" else (1.0 / L)) / n
            comp_recs.append({"update": u + 1, "prompt_index": idx[int(ro["group_ids"][j])], "k": j,
                              "length": int(ro["lengths"][j]), "loss_tokens": t, "truncated": bool(ro["truncated"][j]),
                              "reward": float(ro["rewards"][j].item()), "advantage": a,
                              "per_token_weight": tok_w, "sequence_weight": tok_w * t * abs(a),
                              "response": ro["responses"][j]})
        for rec in comp_recs:
            append_jsonl(comp_path, rec)

        kl_tok = ro["old_logp"] - ro["ref_logp"]
        lens = np.array(ro["lengths"], dtype=float)
        rec = {
            "update": u + 1, "prompt_indices": idx, "loss_type": loss_type,
            "reward_mean": ro["rewards"].mean().item(),
            "group_reward_std_mean": float(np.mean(stds)),
            "uninformative_group_frac": float(np.mean([s <= UNINFORMATIVE_TOL for s in stds])),
            "kl_sampled_token_mean": masked_mean(kl_tok, ro["mask"]).item(),
            "kl_k3_loss_term": diag["sampled_kl"].item(),
            "policy_term": diag["policy_term"].item(), "loss": float(loss.item()),
            "grad_norm": gn, "stepped": stepped,
            "clip_fraction": diag["clip_fraction"].item(), "ratio_mean": diag["ratio_mean"].item(),
            "entropy": masked_mean(ro["entropy"], ro["mask"]).item(),
            "sample_entropy": diag["sample_entropy"].item(),
            "response_length_mean": float(lens.mean()), "response_length_std": float(lens.std()),
            "truncated_rate": float(np.mean(ro["truncated"])),
            "masked_completions": int((T == 0).sum().item()),
            "elapsed_sec": timer(),
            "peak_vram_gib_so_far": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        }
        append_jsonl(log_path, rec)
        print(f"update {u + 1:2d}/{n_updates} [{loss_type}] | reward {rec['reward_mean']:+.3f} | grp-std {rec['group_reward_std_mean']:.3f} | "
              f"uninf {rec['uninformative_group_frac']:.2f} | KL {rec['kl_sampled_token_mean']:+.4f} | gnorm {gn:.3f} | "
              f"len {rec['response_length_mean']:.0f} | trunc {rec['truncated_rate']:.2f} | {rec['elapsed_sec'] / 60:.1f} min", flush=True)
        del ro, adv, token_mask

    policy.save_pretrained(str(out))
    summary = {"run_name": run_name, "loss_type": loss_type, "updates": n_updates, "wall_clock_sec": timer(),
               "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
               "adapter": str(out)}
    save_json(res / f"{run_name}_train_summary.json", summary)
    print(summary, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
