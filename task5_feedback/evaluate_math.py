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
    return {"sft": None, "rlvr": cfg["policies"]["rlvr"], "rlaif": cfg["policies"]["rlaif"]}


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
    wins = sum(p["judge"] == "A" for p in prs)
    ties = sum(p["judge"] == "TIE" for p in prs)
    dec = [p for p in prs if p["verifier"] != "TIE"]
    vt = [p for p in prs if p["verifier"] == "TIE"]
    return {
        "n": n, "wins": wins, "ties": ties, "losses": n - wins - ties,
        "identical_to_sft": sum(p["identical"] for p in prs),
        "win_rate_vs_sft": (wins + 0.5 * ties) / n if n else None,
        "verifier_decisive_pairs": len(dec),
        "judge_agrees_on_decisive": float(np.mean([p["judge"] == p["verifier"] for p in dec])) if dec else None,
        "judge_tie_on_decisive": float(np.mean([p["judge"] == "TIE" for p in dec])) if dec else None,
        "judge_opposite_on_decisive": float(np.mean([p["judge"] not in (p["verifier"], "TIE") for p in dec])) if dec else None,
        "verifier_tie_pairs": len(vt),
        "judge_tie_when_verifier_tie": float(np.mean([p["judge"] == "TIE" for p in vt])) if vt else None,
        "three_way_agreement": float(np.mean([p["judge"] == p["verifier"] for p in prs])) if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg, rows, _ = load_math_evaluation(args.config, args.dataset)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{args.dataset}: {len(rows)} problems", flush=True)

    gens = {p: generate(cfg, p, rows, args.dataset, outdir, args.batch_size) for p in POLICIES}

    judge = PairwiseAIJudge(cfg, outdir / "pairwise_cache.json")
    pairs = []
    for p in ["rlvr", "rlaif"]:
        for a, b in zip(gens[p], gens["sft"]):
            identical = a["response"] == b["response"]
            # A = trained policy, B = SFT. Identical texts are a tie by definition (no judge call).
            pref = "TIE" if identical else judge.compare(a["question"], a["response"], b["response"])
            ver = "A" if a["correct"] > b["correct"] else ("B" if a["correct"] < b["correct"] else "TIE")
            pairs.append({"dataset": args.dataset, "policy": p, "index": a["index"], "judge": pref,
                          "verifier": ver, "identical": identical,
                          "policy_correct": a["correct"], "sft_correct": b["correct"]})
        print(f"[judge] {p} vs sft done", flush=True)
    write_jsonl(outdir / f"pairs_{args.dataset}.jsonl", pairs)

    summary = {
        "dataset": args.dataset, "n_problems": len(rows),
        "decoding": {"greedy": True, "max_new_tokens": int(cfg["math_max_new_tokens"]), "max_prompt_length": 512},
        "policies": {p: policy_summary(gens[p]) for p in POLICIES},
        "pairwise_vs_sft": {p: pair_summary([x for x in pairs if x["policy"] == p]) for p in ["rlvr", "rlaif"]},
        "verifier_judge_agreement_pooled": pair_summary(pairs),
    }
    save_json(outdir / f"eval_{args.dataset}_summary.json", summary)
    for p, s in summary["policies"].items():
        w = summary["pairwise_vs_sft"].get(p, {})
        print(f"{p:6s} acc {s['exact_accuracy']:.3f} | format {s['format_compliance']:.3f} | len {s['length_tokens']['mean']:.0f} "
              f"| trunc {s['truncated_rate']:.2f} | winrate_vs_sft {w.get('win_rate_vs_sft')} | agree {w.get('judge_agrees_on_decisive')}", flush=True)


if __name__ == "__main__":
    main()
