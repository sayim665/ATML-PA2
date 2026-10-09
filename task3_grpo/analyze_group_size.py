from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages

TOL = 1e-6          # manual: informative <=> group reward std > numerical tolerance
LOW_SIGNAL = 0.1    # practical near-tie threshold (reported separately, clearly labelled)


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Split each prompt's first 8 cached completions (in generation_index order) into 8 // k
    disjoint groups of size k. Every K therefore uses all n_prompts * 8 completions: equal total
    generations, with smaller K giving more, smaller groups over the same prompts."""
    if 8 % k:
        raise ValueError(f"K={k} must divide 8")
    groups = []
    for pid, comps in by_prompt.items():
        comps = comps[:8]
        for start in range(0, 8, k):
            groups.append({"prompt": pid, "sub_group": start // k, "rows": comps[start:start + k]})
    return groups


def difficulty_bins(by_prompt):
    """Difficulty = mean reward over all 8 completions; tertiles -> hard / medium / easy."""
    means = {pid: float(np.mean([r["reward"] for r in c[:8]])) for pid, c in by_prompt.items()}
    q1, q2 = np.quantile(list(means.values()), [1 / 3, 2 / 3])
    label = {pid: ("hard" if m <= q1 else "medium" if m <= q2 else "easy") for pid, m in means.items()}
    return label, means, (float(q1), float(q2))


def group_metrics(groups, full_mean):
    rewards = torch.tensor([r["reward"] for g in groups for r in g["rows"]], dtype=torch.float32)
    gids = torch.tensor([i for i, g in enumerate(groups) for _ in g["rows"]])
    adv = group_relative_advantages(rewards, gids)
    stds, centered, base_err, agree = [], [], [], []
    i = 0
    for g in groups:
        r = np.array([x["reward"] for x in g["rows"]], dtype=float)
        stds.append(r.std())
        centered.extend(r - r.mean())
        base_err.append(abs(r.mean() - full_mean[g["prompt"]]))
        for x in g["rows"]:
            a, ref = adv[i].item(), x["reward"] - full_mean[g["prompt"]]
            if abs(a) > 0 and abs(ref) > 1e-9:
                agree.append(bool(np.sign(a) == np.sign(ref)))
            i += 1
    stds = np.array(stds)
    return {
        "n_groups": len(groups), "n_completions": int(rewards.numel()),
        "informative_rate": float((stds > TOL).mean()),
        "uninformative_rate": float((stds <= TOL).mean()),
        "low_signal_rate_std_lt_0p1": float((stds < LOW_SIGNAL).mean()),
        "mean_within_group_std": float(stds.mean()),
        "median_within_group_std": float(np.median(stds)),
        "var_normalized_advantage": float(adv.var(unbiased=False)),
        "var_centered_reward": float(np.var(centered)),
        "nonzero_advantage_rate": float((adv.abs() > 0).float().mean()),
        "baseline_abs_error_vs_k8_mean": float(np.mean(base_err)),
        "sign_agreement_with_k8": float(np.mean(agree)) if agree else None,
        "truncated_rate": float(np.mean([bool(x.get("clipped_at_max", False)) for g in groups for x in g["rows"]])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    by_prompt = load_k8_cache(cfg["group_cache"])
    label, full_mean, cuts = difficulty_bins(by_prompt)
    print(f"Cached prompts: {len(by_prompt)} | completions: {sum(len(c[:8]) for c in by_prompt.values())} | "
          f"difficulty cut points (mean reward): {cuts}")

    rows = []
    for k in [int(x) for x in cfg["group_sizes"]]:
        groups = regroup_equal_generation_budget(by_prompt, k)
        for bin_name in ["all", "hard", "medium", "easy"]:
            sel = groups if bin_name == "all" else [g for g in groups if label[g["prompt"]] == bin_name]
            m = group_metrics(sel, full_mean)
            rows.append({"K": k, "difficulty": bin_name, **m})

    out = {"tolerance": TOL, "low_signal_threshold": LOW_SIGNAL,
           "regrouping": "each prompt's 8 cached completions (generation_index order) split into 8//K disjoint groups; "
                         "all K use the same 192 completions",
           "difficulty_rule": "prompt mean reward over its 8 cached completions; tertiles -> hard/medium/easy",
           "difficulty_cut_points": cuts,
           "prompt_difficulty": {pid: {"mean_reward": full_mean[pid], "bin": label[pid]} for pid in full_mean},
           "results": rows}
    save_json(res / "group_size_study.json", out)
    with open(res / "group_size_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'K':>2} {'bin':7s} {'groups':>6} {'inform':>7} {'lowsig':>7} {'grp_std':>8} {'var_adv':>8} "
          f"{'var_ctr':>8} {'base_err':>8} {'sign_agr':>8}")
    for r in rows:
        sa = f"{r['sign_agreement_with_k8']:.3f}" if r["sign_agreement_with_k8"] is not None else "  n/a"
        print(f"{r['K']:>2} {r['difficulty']:7s} {r['n_groups']:>6} {r['informative_rate']:>7.3f} "
              f"{r['low_signal_rate_std_lt_0p1']:>7.3f} {r['mean_within_group_std']:>8.3f} {r['var_normalized_advantage']:>8.3f} "
              f"{r['var_centered_reward']:>8.3f} {r['baseline_abs_error_vs_k8_mean']:>8.3f} {sa:>8}")


if __name__ == "__main__":
    main()
