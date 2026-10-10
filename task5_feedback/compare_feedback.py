from __future__ import annotations

import argparse
import csv

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm, tr, diag = load_json(d / "eval_gsm_summary.json"), load_json(d / "eval_transfer_summary.json"), load_json(d / "diagnostics_summary.json")

    rows = []
    for p in ["sft", "rlvr", "rlaif"]:
        g, t = gsm["policies"][p], tr["policies"][p]
        gw, tw = gsm["pairwise_vs_sft"].get(p, {}), tr["pairwise_vs_sft"].get(p, {})
        rows.append({
            "policy": p,
            "gsm_accuracy": g["exact_accuracy"], "svamp_accuracy": t["exact_accuracy"],
            "accuracy_drop": g["exact_accuracy"] - t["exact_accuracy"],
            "gsm_format": g["format_compliance"], "svamp_format": t["format_compliance"],
            "gsm_winrate_vs_sft": gw.get("win_rate_vs_sft"), "svamp_winrate_vs_sft": tw.get("win_rate_vs_sft"),
            "winrate_drop": (gw["win_rate_vs_sft"] - tw["win_rate_vs_sft"]) if gw and tw else None,
            "gsm_len_mean": g["length_tokens"]["mean"], "svamp_len_mean": t["length_tokens"]["mean"],
            "gsm_truncated": g["truncated_rate"], "svamp_truncated": t["truncated_rate"],
            "gsm_judge_agrees_on_decisive": gw.get("judge_agrees_on_decisive"),
            "svamp_judge_agrees_on_decisive": tw.get("judge_agrees_on_decisive"),
        })
    out = {"policies": rows,
           "verifier_judge_agreement": {"gsm": gsm["verifier_judge_agreement_pooled"],
                                        "transfer": tr["verifier_judge_agreement_pooled"]},
           "diagnostics": {"S_reason": diag["S_reason"], "S_outcome": diag["S_outcome"],
                           "by_category": diag["by_category"], "per_variant_reward_means": diag["per_variant_reward_means"]},
           "failure_types": {"gsm": {p: gsm["policies"][p]["failure_types"] for p in gsm["policies"]},
                             "transfer": {p: tr["policies"][p]["failure_types"] for p in tr["policies"]}}}
    save_json(d / "task5_comparison.json", out)
    with open(d / "task5_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("S_reason:", diag["S_reason"], "| S_outcome:", diag["S_outcome"])


if __name__ == "__main__":
    main()
