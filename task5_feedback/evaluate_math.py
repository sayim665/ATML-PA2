from __future__ import annotations

import argparse
import gc
import re

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

POLICIES = ["sft", "rlvr", "rlaif"]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def summarize(xs):
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {"n": 0}
    q1, q3 = np.percentile(a, [25, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)),
            "iqr": float(q3 - q1), "n": int(a.size)}


def gold_mentioned(response: str, gold: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(gold)}(?![\d.])", response.replace(",", "")) is not None


def generate(cfg, name, rows, dataset, outdir, batch_size):
    path = outdir / f"gen_{dataset}_{name}.jsonl"
    if path.exists():
        print(f"[skip] {path.name} exists", flush=True)
        return read_jsonl(path)
    tok = load_tokenizer(cfg["base_model"])
    model = load_frozen_policy(cfg, name)
    set_seed(int(cfg["seed"]))
    max_new, timer, recs = int(cfg["math_max_new_tokens"]), wall_timer(), []
    for s in range(0, len(rows), batch_size):
        batch = rows[s:s + batch_size]
        gen = batch_generate(model, tok, [prompt_messages(r) for r in batch], max_prompt_length=512,
                             max_new_tokens=max_new, temperature=0.0, top_p=1.0, do_sample=False)
        for j, r in enumerate(batch):
            resp, gold = gen["responses"][j], str(r["gold_final"])
            pred = extract_designated_final(resp)
            recs.append({"index": s + j, "id": r.get("prompt_id", r.get("source_index")), "question": r["question"],
                         "gold_final": gold, "policy": name, "response": resp, "pred": pred,
                         "format_ok": pred is not None, "correct": exact_reward(resp, gold),
                         "length_tokens": gen["response_lengths"][j], "truncated": gen["truncated"][j]})
        print(f"[{dataset}/{name}] {min(s + batch_size, len(rows))}/{len(rows)}  {timer() / 60:.1f} min", flush=True)
    write_jsonl(path, recs)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return recs


def policy_summary(recs):
    wrong = [r for r in recs if not r["correct"]]
    return {
        "n": len(recs),
        "exact_accuracy": float(np.mean([r["correct"] for r in recs])),
        "format_compliance": float(np.mean([r["format_ok"] for r in recs])),
        "length_tokens": summarize([r["length_tokens"] for r in recs]),
        "truncated_rate": float(np.mean([r["truncated"] for r in recs])),
        "failure_types": {
            "n_wrong": len(wrong),
            "no_final_answer": sum(not r["format_ok"] for r in wrong),
            "no_final_answer_truncated": sum((not r["format_ok"]) and r["truncated"] for r in wrong),
            "wrong_final_number": sum(r["format_ok"] for r in wrong),
            "gold_mentioned_but_wrong_final": sum(r["format_ok"] and gold_mentioned(r["response"], r["gold_final"]) for r in wrong),
        },
    }


def pair_summary(prs):
    n = len(prs)
