from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from common.data import repo_path
from common.logging_utils import load_json


def tag(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def fork_name(eps: float, kl: float) -> str:
    return f"fork_eps{tag(eps)}_kl{tag(kl)}"


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_fork(config: str, cfg: dict, eps: float, kl: float) -> str:
    """Train (if needed) and evaluate (if needed) one short fork from the identical PPO midpoint."""
    name, res = fork_name(eps, kl), repo_path(cfg["results_dir"])
    out = f"outputs/task2_ppo/{name}"
    if not (res / f"{name}_train_summary.json").exists():
        run([sys.executable, "-u", "-m", "task2_ppo.continue_train", "--config", config, "--run-name", name,
             "--updates", str(int(cfg["fork_updates"])), "--clip-epsilon", str(eps), "--kl-beta", str(kl),
             "--output", out])
    if not (res / f"eval_{name}_summary.json").exists():
        run([sys.executable, "-u", "-m", "task2_ppo.evaluate", "--config", config, "--adapter", out, "--name", name])
    return name


def read_log(cfg, name):
    return [json.loads(l) for l in open(repo_path(cfg["results_dir"]) / f"{name}_train_log.jsonl")]


def fork_row(cfg, name, eps, kl):
    res = repo_path(cfg["results_dir"])
    log, ev, tr = read_log(cfg, name), load_json(res / f"eval_{name}_summary.json"), load_json(res / f"{name}_train_summary.json")
    step_kl = [abs(r["approx_kl_old_last_epoch"]) for r in log]
    return {
        "run": name, "clip_epsilon": eps, "kl_beta": kl, "updates": len(log),
        # stability statistic: per-update policy step size (approx KL rollout-policy -> once-updated policy)
        "stability_step_kl_mean": float(np.mean(step_kl)), "stability_step_kl_max": float(np.max(step_kl)),
        "policy_grad_norm_mean": float(np.mean([r["policy_grad_norm"] for r in log])),
        "policy_grad_norm_std": float(np.std([r["policy_grad_norm"] for r in log])),
        "train_clip_fraction_last_epoch_mean": float(np.mean([r["clip_fraction_last_epoch"] for r in log])),
        "train_ratio_max": float(max(r["ratio_max"] for r in log)),
        "train_ratio_min": float(min(r["ratio_min"] for r in log)),
        "value_loss_mean": float(np.mean([r["value_loss"] for r in log])),
        "value_ev_mean": float(np.nanmean([r["value_explained_variance"] for r in log])),
        "skipped_steps": int(sum(r["skipped_steps"] for r in log)),
        "train_reward_first": log[0]["reward_effective"], "train_reward_last": log[-1]["reward_effective"],
        "heldout_reward_effective": ev["reward_effective"]["mean"], "heldout_reward_effective_std": ev["reward_effective"]["std"],
        "heldout_reward_raw": ev["reward_raw"]["mean"],
        "heldout_kl_token_mean": ev["kl_token_mean"], "heldout_entropy": ev["entropy_token_mean"],
        "heldout_length_mean": ev["length_tokens"]["mean"], "heldout_length_std": ev["length_tokens"]["std"],
        "heldout_truncated_rate": ev["truncated_rate"],
        "train_wall_clock_sec": tr["wall_clock_sec"], "train_peak_vram_gib": tr["peak_vram_gib"],
    }


def trajectories(cfg, name):
    log = read_log(cfg, name)
    keys = ["reward_effective", "kl_token_mean", "entropy", "response_length", "policy_loss", "value_loss",
            "clip_fraction_last_epoch", "approx_kl_old_last_epoch", "policy_grad_norm", "value_explained_variance"]
    return {k: [r[k] for r in log] for k in keys}
