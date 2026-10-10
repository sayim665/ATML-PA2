from __future__ import annotations

import argparse
import gc

import pandas as pd
import torch

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", default="sft,dpo,ppo,grpo")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_xstest(cfg)
    print("XSTest rows:", len(df), "| by class:", df["benchmark_class"].value_counts().to_dict(), flush=True)

    for name in args.policies.split(","):
        path = outdir / f"generated_{name}.jsonl"
        if path.exists():
            print(f"[skip] {path} already exists", flush=True)
            continue
        adapter = policy_specs(cfg)[name]
        if adapter is not None and not repo_path(adapter).exists():
            print(f"[missing] adapter for {name}: {adapter} - run its standard training first", flush=True)
            continue
        set_seed(int(cfg["seed"]))
        timer = wall_timer()
        recs = generate_for_policy(cfg, name, batch_size=args.batch_size)   # fixed CSV order preserved
        write_jsonl(path, recs)
        save_json(outdir / f"generated_{name}_meta.json", {
            "policy": name, "adapter": adapter, "n": len(recs), "decoding": "greedy (do_sample=False)",
            "max_new_tokens": int(cfg["safety_max_new_tokens"]), "max_prompt_length": 256,
            "batch_size": args.batch_size, "seconds": timer()})
        print(f"[done] {name}: {len(recs)} responses in {timer() / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
