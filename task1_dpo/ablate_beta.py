from __future__ import annotations

import argparse
import csv
import subprocess
import sys

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def tag(beta: float) -> str:
    return "beta_" + f"{beta:g}".replace(".", "p")          # 0.03 -> beta_0p03


def collect(res, name, beta, budget):
    tr = load_json(res / f"{name}_train_summary.json")
    ev = load_json(res / f"eval_{name}_summary.json")
    cfg_run = load_json(res / f"{name}_config.json")
    log = read_jsonl(res / f"{name}_train_log.jsonl")
    tail = log[-6:-1] if len(log) > 5 else log
    return {
        "run": name, "beta": beta, "budget": budget,
        "train_examples": cfg_run["num_examples"], "optimizer_steps": tr["optimizer_steps"],
        "final_train_loss_last5": float(np.mean([r["loss"] for r in tail])),
        "final_train_acc_last5": float(np.mean([r["pref_acc"] for r in tail])),
        "heldout_dpo_loss": ev["heldout_dpo_loss"],
        "heldout_pref_accuracy": ev["heldout_pref_accuracy"],
        "heldout_margin_mean": ev["heldout_margin"]["mean"],
        "kl_token_mean": ev["kl_token_mean"],
        "kl_seq_sum_mean": ev["kl_seq_sum"]["mean"],
        "reward_mean": ev["reward"]["mean"], "reward_std": ev["reward"]["std"],
        "length_mean": ev["length_tokens"]["mean"], "length_std": ev["length_tokens"]["std"],
        "length_iqr": ev["length_tokens"]["iqr"], "truncated_rate": ev["truncated_rate"],
        "train_wall_clock_sec": tr["wall_clock_sec"], "train_peak_vram_gib": tr["peak_vram_gib"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--force", action="store_true", help="re-run even if results already exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    n = int(cfg["short_ablation_examples"])
    print("beta values:", cfg["betas"], "| examples per fork:", n, flush=True)

    rows = []
    for beta in [float(b) for b in cfg["betas"]]:
        name, out = tag(beta), f"outputs/task1_dpo/{tag(beta)}"
        if args.force or not (res / f"{name}_train_summary.json").exists():
            run([sys.executable, "-u", "-m", "task1_dpo.train", "--config", args.config, "--run-name", name,
                 "--beta", str(beta), "--max-examples", str(n), "--output", out])
        if args.force or not (res / f"eval_{name}_summary.json").exists():
            run([sys.executable, "-u", "-m", "task1_dpo.evaluate", "--config", args.config, "--adapter", out,
                 "--name", name, "--beta", str(beta), "--skip-word-limit"])
        rows.append(collect(res, name, beta, f"short fork: first {n} valid pairs, 1 pass"))

    if (res / "eval_standard_summary.json").exists():
        rows.append(collect(res, "standard", float(cfg["beta"]), "standard: full set, 1 epoch"))

    save_json(res / "beta_sweep.json", rows)
    with open(res / "beta_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['run']:12s} beta={r['beta']:<5} acc={r['heldout_pref_accuracy']:.3f} "
              f"margin={r['heldout_margin_mean']:+.3f} KL/tok={r['kl_token_mean']:+.2e} "
              f"reward={r['reward_mean']:.3f} len={r['length_mean']:.1f}")


if __name__ == "__main__":
    main()
