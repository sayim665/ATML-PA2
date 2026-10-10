"""Secondary analyses cited in the report (all CPU, all read saved result files).
Run: python -m scripts.extra_analyses
"""
from __future__ import annotations

import numpy as np

from common.data import prompt_messages, read_jsonl
from common.logging_utils import save_json
from common.models import load_tokenizer


def dpo_vs_sft_paired():
    dpo = {r["index"]: r for r in read_jsonl("results/task1_dpo/eval_standard_generations.jsonl")}
    sft = {r["index"]: r for r in read_jsonl("results/task1_dpo/eval_sft_generations.jsonl")}
    idx = sorted(dpo)
    dr = np.array([dpo[i]["reward"] - sft[i]["reward"] for i in idx])
    dl = np.array([dpo[i]["length_tokens"] - sft[i]["length_tokens"] for i in idx])
    rng = np.random.default_rng(6304)
    boot = [rng.choice(dr, len(dr)).mean() for _ in range(5000)]
    return {"n_prompts": len(idx),
            "identical_response_rate": float(np.mean([dpo[i]["response"] == sft[i]["response"] for i in idx])),
            "reward_diff_mean": float(dr.mean()),
            "reward_diff_95ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "dpo_higher_reward_rate": float(np.mean(dr > 0)),
            "length_diff_mean": float(dl.mean()), "length_diff_std": float(dl.std())}


def likelihood_displacement():
    out = {}
    for name in ["standard_strat", "length_balanced_strat"]:
        P = read_jsonl(f"results/task1_dpo/eval_{name}_pairs.jsonl")
        out[name] = {s: {"n": len(g),
                         "chosen_logratio_mean": float(np.mean([p["chosen_logratio"] for p in g])),
                         "rejected_logratio_mean": float(np.mean([p["rejected_logratio"] for p in g])),
                         "margin_mean": float(np.mean([p["margin"] for p in g]))}
                     for s in ["preferred_longer", "length_matched", "rejected_longer"]
                     for g in [[p for p in P if p["stratum"] == s]]}
    return out


def prompt_length_check():
    tok = load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    plen = lambda r: len(tok.apply_chat_template(prompt_messages(r), tokenize=True, add_generation_prompt=True))
    tr = np.array([plen(r) for r in read_jsonl("data/rl_prompt_pool_train.jsonl")])
    ev = np.array([plen(r) for r in read_jsonl("data/rl_prompt_pool_eval.jsonl")[:32]])
    g18 = [r for r in read_jsonl("results/task3_grpo/standard_completions.jsonl") if r["update"] == 18]
    rows = read_jsonl("data/rl_prompt_pool_train.jsonl")
    return {"generation_prompt_cap": 256, "reward_window_grpo": 1024, "reward_window_ppo": 1280,
            "train_frac_over_256": float((tr > 256).mean()), "train_frac_over_1024": float((tr > 1024).mean()),
            "train_frac_over_1280": float((tr > 1280).mean()),
            "eval32_over_256": int((ev > 256).sum()), "eval32_over_256_indices": np.where(ev > 256)[0].tolist(),
            "grpo_update18_prompt_tokens": plen(rows[g18[0]["prompt_index"]]),
            "grpo_update18_rewards": [r["reward"] for r in g18],
            "grpo_update18_identical_texts": len({r["response"] for r in g18}) == 1}


def ppo_midpoint_vs_standard():
    mid = {r["index"]: r for r in read_jsonl("results/task2_ppo/eval_midpoint_generations.jsonl")}
    std = {r["index"]: r for r in read_jsonl("results/task2_ppo/eval_standard_generations.jsonl")}
    return {"n": len(mid), "identical_responses": sum(mid[i]["response"] == std[i]["response"] for i in mid)}


def main():
    save_json("results/task1_dpo/standard_vs_sft_paired.json", dpo_vs_sft_paired())
    save_json("results/task1_dpo/likelihood_displacement.json", likelihood_displacement())
    save_json("results/pipeline_prompt_length_check.json", prompt_length_check())
    save_json("results/task2_ppo/midpoint_vs_standard_identical.json", ppo_midpoint_vs_standard())
    print("wrote: standard_vs_sft_paired, likelihood_displacement, pipeline_prompt_length_check, midpoint_vs_standard_identical")


if __name__ == "__main__":
    main()
