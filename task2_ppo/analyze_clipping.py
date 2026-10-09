from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json
from common.metrics import masked_mean
from common.models import load_policy, load_tokenizer, reference_mode
from task2_ppo.continue_train import policy_token_stats
from task2_ppo.forks import ensure_fork, fork_row
from task2_ppo.ppo import compute_gae, normalize_advantages, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)
    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


@torch.no_grad()
def cached_batch_study(cfg):
    """Fixed cached batch: ratio of the supplied midpoint policy vs cached old log-probs, per epsilon."""
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    device = next(policy.parameters()).device
    pool = {r.get("prompt_id"): r for r in read_jsonl(cfg["paths"]["rl_prompt_train"]) + read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    rows = load_cached_rollouts(cfg["cached_rollouts"])

    per_tok = {"ratio": [], "adv": [], "mask": []}
    align = {"exact_length_match": 0, "n": len(rows), "ref_abs_diff_mean": [], "old_abs_diff_mean": []}
    for row in rows:
        msgs = prompt_messages(pool[row["prompt_id"]])
        prompt_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        resp_ids = tok(row["response"], add_special_tokens=False)["input_ids"]
        T = len(row["old_logprobs"])
        if len(resp_ids) == T - 1 and row.get("terminated_with_eos", False):
            resp_ids = resp_ids + [tok.eos_token_id]
        align["exact_length_match"] += int(len(resp_ids) == T)
        n = min(len(resp_ids), T)
        seq = torch.tensor([prompt_ids + resp_ids[:n]], device=device)
        attn = torch.ones_like(seq)
        resp = torch.tensor([resp_ids[:n]], device=device)
        new_lp, _ = policy_token_stats(policy, seq, attn, len(prompt_ids), resp)
        with reference_mode(policy):
            ref_lp, _ = policy_token_stats(policy, seq, attn, len(prompt_ids), resp)
        old = row["old_logprobs"][:n].float().to(device)[None]
        ref_c = row["ref_logprobs"][:n].float().to(device)[None]
        vals = row["values"][:n].float().to(device)[None]
        mask = torch.ones_like(old)
        r_eff = torch.tensor([float(row.get("effective_terminal_reward", row.get("raw_terminal_reward", 0.0)))], device=device)
        align["ref_abs_diff_mean"].append((ref_lp - ref_c).abs().mean().item())
        align["old_abs_diff_mean"].append((new_lp - old).abs().mean().item())
        rew = shaped_rewards(r_eff, old, ref_c, mask, float(cfg["kl_beta"]))
        adv_raw, _ = compute_gae(rew, vals, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
        per_tok["ratio"].append(torch.exp(new_lp - old)[0].cpu())
        per_tok["adv"].append(adv_raw[0].cpu())
        per_tok["mask"].append(mask[0].cpu())

    ratio, adv_raw, mask = (torch.cat(per_tok[k])[None] for k in ("ratio", "adv", "mask"))
    adv = normalize_advantages(adv_raw, mask)          # batch-level whitening, as in the training loop
    out = {"n_rollouts": len(rows), "n_tokens": int(mask.sum().item()),
           "alignment": {"exact_length_match": align["exact_length_match"], "n": align["n"],
                         "ref_logprob_abs_diff_mean": float(np.mean(align["ref_abs_diff_mean"])),
                         "new_vs_old_logprob_abs_diff_mean": float(np.mean(align["old_abs_diff_mean"]))},
           "ratio_stats": {"mean": ratio.mean().item(), "std": ratio.std().item(), "min": ratio.min().item(),
                           "max": ratio.max().item(), "p05": float(np.percentile(ratio.numpy(), 5)),
                           "p95": float(np.percentile(ratio.numpy(), 95))},
           "unclipped_surrogate": masked_mean(ratio * adv, mask).item(), "per_epsilon": []}
    for eps in [float(e) for e in cfg["clip_values"]]:
        clipped = ratio.clamp(1 - eps, 1 + eps)
        surr = torch.minimum(ratio * adv, clipped * adv)
        outside = (ratio < 1 - eps) | (ratio > 1 + eps)
        binding = ((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))
        out["per_epsilon"].append({
            "epsilon": eps,
            "clip_fraction_outside_band": masked_mean(outside.float(), mask).item(),
            "affected_token_fraction_binding": masked_mean(binding.float(), mask).item(),
            "clipped_surrogate": masked_mean(surr, mask).item(),
            "surrogate_reduction_vs_unclipped": (masked_mean(ratio * adv, mask) - masked_mean(surr, mask)).item(),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)

    if not (res / "clipping_cached_batch.json").exists():
        cached = cached_batch_study(cfg)
        save_json(res / "clipping_cached_batch.json", cached)
        print(cached, flush=True)
        torch.cuda.empty_cache()

    kl = float(cfg["kl_beta"])
    rows = [fork_row(cfg, ensure_fork(args.config, cfg, float(e), kl), float(e), kl) for e in cfg["clip_values"]]
    save_json(res / "clipping_forks.json", rows)
    with open(res / "clipping_forks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
