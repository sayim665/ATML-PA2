from __future__ import annotations

import argparse
import csv

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task2_ppo.forks import ensure_fork, fork_row, trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    eps = float(cfg["clip_epsilon"])
    rows, traj = [], {}
    for kl in [float(b) for b in cfg["kl_values"]]:
        name = ensure_fork(args.config, cfg, eps, kl)
        rows.append(fork_row(cfg, name, eps, kl))
        traj[name] = trajectories(cfg, name)
    save_json(res / "kl_forks.json", {"rows": rows, "trajectories": traj})
    with open(res / "kl_forks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
