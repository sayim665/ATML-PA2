from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json

RUNS = [("norm_grpo", "grpo"), ("norm_dr_grpo", "dr_grpo")]


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def allocation(comps, cuts):
    used = [c for c in comps if c["loss_tokens"] > 0]
    total = sum(c["sequence_weight"] for c in used) or 1.0

    def bin_of(L):
        return "short" if L <= cuts[0] else ("medium" if L <= cuts[1] else "long")

    out = {}
    for b in ["short", "medium", "long"]:
        sel = [c for c in used if bin_of(c["length"]) == b]
        out[b] = {
            "n": len(sel),
            "mean_length": float(np.mean([c["length"] for c in sel])) if sel else None,
            "mean_per_token_weight": float(np.mean([c["per_token_weight"] for c in sel])) if sel else None,
            "mean_sequence_weight": float(np.mean([c["sequence_weight"] for c in sel])) if sel else None,
            "share_of_total_weight": float(sum(c["sequence_weight"] for c in sel) / total),
            "mean_seq_weight_pos_adv": float(np.mean([c["sequence_weight"] for c in sel if c["advantage"] > 0])) if any(c["advantage"] > 0 for c in sel) else None,
            "mean_seq_weight_neg_adv": float(np.mean([c["sequence_weight"] for c in sel if c["advantage"] < 0])) if any(c["advantage"] < 0 for c in sel) else None,
        }
    lens = np.array([c["length"] for c in used], float)
    w = np.array([c["sequence_weight"] for c in used], float)
    out["corr_length_vs_sequence_weight"] = float(np.corrcoef(lens, w)[0, 1]) if len(used) > 2 and w.std() > 0 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    upd = str(int(cfg["fork_updates"]))

    for name, lt in RUNS:
        out = f"outputs/task3_grpo/{name}"
        if not (res / f"{name}_train_summary.json").exists():
            run([sys.executable, "-u", "-m", "task3_grpo.continue_train", "--config", args.config, "--run-name", name,
                 "--updates", upd, "--loss-type", lt, "--output", out])
        if not (res / f"eval_{name}_summary.json").exists():
            run([sys.executable, "-u", "-m", "task3_grpo.evaluate", "--config", args.config, "--adapter", out, "--name", name])

    comps = {name: load_jsonl(res / f"{name}_completions.jsonl") for name, _ in RUNS}
    logs = {name: load_jsonl(res / f"{name}_train_log.jsonl") for name, _ in RUNS}
    pooled = [c["length"] for name in comps for c in comps[name] if c["loss_tokens"] > 0]
    cuts = [float(x) for x in np.quantile(pooled, [1 / 3, 2 / 3])]

    # controlled check: update-1 completions are identical across forks (same start + seed)
    u1 = {name: [c for c in comps[name] if c["update"] == 1] for name, _ in RUNS}
    a, b = u1["norm_grpo"], u1["norm_dr_grpo"]
    same = len(a) == len(b) and all(x["response"] == y["response"] for x, y in zip(a, b))
    controlled = [{"length": x["length"], "advantage": x["advantage"], "loss_tokens": x["loss_tokens"],
                   "grpo_per_token_weight": x["per_token_weight"], "dr_grpo_per_token_weight": y["per_token_weight"],
                   "grpo_sequence_weight": x["sequence_weight"], "dr_grpo_sequence_weight": y["sequence_weight"]}
                  for x, y in zip(a, b)]

    rows = {}
    for name, lt in RUNS:
        ev, lg = load_json(res / f"eval_{name}_summary.json"), logs[name]
        rows[name] = {
            "loss_type": lt, "updates": len(lg),
            "train_length_first2": float(np.mean([r["response_length_mean"] for r in lg[:2]])),
            "train_length_last2": float(np.mean([r["response_length_mean"] for r in lg[-2:]])),
            "train_truncated_rate_mean": float(np.mean([r["truncated_rate"] for r in lg])),
            "train_grad_norm_mean": float(np.nanmean([r["grad_norm"] for r in lg])),
            "train_reward_mean": float(np.mean([r["reward_mean"] for r in lg])),
            "heldout_reward": ev["reward"]["mean"], "heldout_reward_std": ev["reward"]["std"],
            "heldout_kl_token_mean": ev["kl_token_mean"], "heldout_entropy": ev["entropy_token_mean"],
            "heldout_length_mean": ev["length_tokens"]["mean"], "heldout_length_std": ev["length_tokens"]["std"],
            "heldout_truncated_rate": ev["truncated_rate"],
            "allocation_by_length": allocation(comps[name], cuts),
            "length_trajectory": [r["response_length_mean"] for r in lg],
        }
    out = {"length_bin_cut_points_tokens": cuts, "update1_identical_samples": same,
           "update1_controlled_weights": controlled, "runs": rows}
    save_json(res / "normalization_comparison.json", out)
    for name in rows:
        r = rows[name]
        print(name, {k: v for k, v in r.items() if k not in ("allocation_by_length", "length_trajectory")}, flush=True)
        print("  allocation:", json.dumps(r["allocation_by_length"]), flush=True)


if __name__ == "__main__":
    main()
