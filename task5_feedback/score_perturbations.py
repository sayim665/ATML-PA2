from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
# clean first, so the round-robin's (clean, other) comparisons are the same cached calls as the controlled pairs
ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
         "persuasive_filler_correct", "gold_distractor_wrong_final"]
# controlled pairs: (category, perturbed variant); the diagnostically better response is always clean_correct
PAIRS = [("reasoning", "corrupt_reasoning_correct_final"),
         ("outcome", "good_reasoning_wrong_final"),
         ("filler", "persuasive_filler_correct"),
         ("distractor", "gold_distractor_wrong_final")]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def rates(xs):
    n = len(xs)
    return {"n": n, "better_rate": xs.count("better") / n, "tie_rate": xs.count("tie") / n,
            "wrong_preference_rate": xs.count("wrong") / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups), flush=True)
    judge = PairwiseAIJudge(cfg, outdir / "pairwise_cache.json")

    per_variant, pairs = [], []
    for pid, vs in groups.items():
        q, gold = vs["clean_correct"]["question"], str(vs["clean_correct"]["gold_final"])
        resp = {v: vs[v]["response"] for v in ORDER}
        group_reward = judge.group_rewards(q, [resp[v] for v in ORDER])   # direct-RLAIF reward, as in training
        for v, g in zip(ORDER, group_reward):
            per_variant.append({"problem_id": pid, "variant": v, "verifier_reward": exact_reward(resp[v], gold),
                                "expected_exact_reward": vs[v].get("expected_exact_reward"), "rlaif_group_reward": g})
        for cat, other in PAIRS:
            rc, ro = exact_reward(resp["clean_correct"], gold), exact_reward(resp[other], gold)
            ver = "better" if rc > ro else ("tie" if rc == ro else "wrong")
            pref = judge.compare(q, resp["clean_correct"], resp[other])     # A = clean (better)
            jud = {"A": "better", "TIE": "tie", "B": "wrong"}[pref]
            pairs.append({"problem_id": pid, "category": cat, "perturbed_variant": other,
                          "verifier": ver, "judge": jud, "judge_raw": pref})
        print(f"[diag] problem {pid} done", flush=True)

    write_jsonl(outdir / "diagnostic_pairs.jsonl", pairs)
    write_jsonl(outdir / "diagnostic_variant_rewards.jsonl", per_variant)
    by_cat = {cat: {mech: rates([p[mech] for p in pairs if p["category"] == cat]) for mech in ["verifier", "judge"]}
              for cat, _ in PAIRS}
    variant_means = {v: {"verifier_reward_mean": float(np.mean([r["verifier_reward"] for r in per_variant if r["variant"] == v])),
                         "rlaif_group_reward_mean": float(np.mean([r["rlaif_group_reward"] for r in per_variant if r["variant"] == v]))}
                     for v in ORDER}
    mismatch = [r for r in per_variant if r["expected_exact_reward"] is not None
                and float(r["verifier_reward"]) != float(r["expected_exact_reward"])]
    summary = {
        "n_problems": len(groups),
        "pair_definition": "each perturbed variant vs clean_correct; clean_correct is the diagnostically better response",
        "by_category": by_cat,
        "S_reason": {m: by_cat["reasoning"][m]["better_rate"] for m in ["verifier", "judge"]},
        "S_outcome": {m: by_cat["outcome"][m]["better_rate"] for m in ["verifier", "judge"]},
        "per_variant_reward_means": variant_means,
        "verifier_vs_expected_exact_reward_mismatches": len(mismatch),
    }
    save_json(outdir / "diagnostics_summary.json", summary)
    for cat, d in by_cat.items():
        for m, r in d.items():
            print(f"{cat:10s} {m:8s} better {r['better_rate']:.2f} | tie {r['tie_rate']:.2f} | wrong {r['wrong_preference_rate']:.2f}")
    print("S_reason:", summary["S_reason"], "| S_outcome:", summary["S_outcome"],
          "| verifier mismatches vs expected:", len(mismatch))


if __name__ == "__main__":
    main()
