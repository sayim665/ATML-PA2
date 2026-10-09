from __future__ import annotations

import argparse
import csv
import subprocess
import sys

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.models import load_tokenizer

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def dataset_length_profile(path, tokenizer):
    """Length structure of the PREFERENCE DATA itself (not the policy)."""
    rows = read_jsonl(path)
    lc, lr = [], []
    for r in rows:
        yc, yr = preference_responses(r)
        lc.append(len(tokenizer(yc, add_special_tokens=False)["input_ids"]))
        lr.append(len(tokenizer(yr, add_special_tokens=False)["input_ids"]))
    lc, lr = np.array(lc), np.array(lr)
    d = lc - lr
    return {
        "file": str(path), "n_pairs": len(rows),
        "chosen_tokens_mean": float(lc.mean()), "rejected_tokens_mean": float(lr.mean()),
        "mean_chosen_minus_rejected": float(d.mean()),
        "frac_chosen_longer": float((d > 0).mean()), "frac_rejected_longer": float((d < 0).mean()),
        "frac_equal": float((d == 0).mean()),
        "corr_length_diff_with_preference_sign": float(np.corrcoef(d, np.ones_like(d))[0, 1]) if d.std() > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    std_train, bal_train = cfg["paths"]["dpo_standard_train"], cfg["paths"]["dpo_length_train"]
    strat_eval = cfg["paths"]["dpo_length_eval"]
    bal_out = cfg["length_output"]

    # 1) dataset-level length structure
    tok = load_tokenizer(cfg["base_model"])
    profile = {"standard_train": dataset_length_profile(std_train, tok),
               "length_balanced_train": dataset_length_profile(bal_train, tok)}
    for p in profile.values():
        p.pop("corr_length_diff_with_preference_sign", None)
    save_json(res / "length_dataset_profile.json", profile)
    print(profile, flush=True)

    # 2) train the length-balanced model (same init / beta / optimizer / seed / LoRA as standard)
    if args.force or not (res / "length_balanced_train_summary.json").exists():
        run([sys.executable, "-u", "-m", "task1_dpo.train", "--config", args.config,
             "--run-name", "length_balanced", "--dataset", bal_train, "--output", bal_out])

    # 3) both models on the length-stratified held-out set (preference metrics only)
    for name, adapter in [("standard", cfg["standard_output"]), ("length_balanced", bal_out)]:
        if args.force or not (res / f"eval_{name}_strat_summary.json").exists():
            run([sys.executable, "-u", "-m", "task1_dpo.evaluate", "--config", args.config, "--adapter", adapter,
                 "--name", f"{name}_strat", "--pairs", strat_eval, "--skip-generation", "--skip-word-limit"])

    # 4) balanced model: generation length + word-limit compliance on the SAME common prompts as standard
    if args.force or not (res / "eval_length_balanced_summary.json").exists():
        run([sys.executable, "-u", "-m", "task1_dpo.evaluate", "--config", args.config,
             "--adapter", bal_out, "--name", "length_balanced"])

    # 5) comparison table
    rows = []
    for name in ["sft", "standard", "length_balanced"]:
        gen = load_json(res / f"eval_{name}_summary.json")
        strat = load_json(res / f"eval_{name}_strat_summary.json") if name != "sft" else None
        row = {"model": name}
        if strat:
            row["strat_overall_acc"] = strat["heldout_pref_accuracy"]
            for s in STRATA:
                b = strat["by_stratum"].get(s, {})
                row[f"acc_{s}"] = b.get("pref_accuracy")
                row[f"n_{s}"] = b.get("n")
                row[f"margin_{s}"] = b.get("margin_mean")
        row.update({
            "std_heldout_acc": gen["heldout_pref_accuracy"],
            "gen_len_mean": gen["length_tokens"]["mean"], "gen_len_std": gen["length_tokens"]["std"],
            "gen_len_iqr": gen["length_tokens"]["iqr"], "truncated_rate": gen["truncated_rate"],
            "reward_mean": gen["reward"]["mean"], "kl_token_mean": gen["kl_token_mean"],
            "word_limit_compliance": gen.get("word_limit_compliance"),
            "word_limit_words_mean": gen.get("word_limit_word_count", {}).get("mean"),
        })
        rows.append(row)

    save_json(res / "length_study.json", {"dataset_profile": profile, "models": rows})
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "model", k))
    with open(res / "length_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
