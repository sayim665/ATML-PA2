from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task2_ppo.continue_train import policy_token_stats


def summarize(xs):
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {"n": 0}
    q1, q3 = np.percentile(a, [25, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)),
            "iqr": float(q3 - q1), "n": int(a.size)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the untouched SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--num-prompts", type=int, default=32, help="course acceptance protocol used 32 held-out prompts")
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, help="default: eval_max_response_length from config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    max_new = int(args.max_new or cfg["eval_max_response_length"])
    res = repo_path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    timer = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tok = load_tokenizer(cfg["base_model"])
    adapter = None if args.adapter.lower() == "none" else args.adapter
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg)
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])[: args.num_prompts]
    g, pen = cfg["generation"], float(cfg.get("missing_eos_penalty", 0.0))

    set_seed(seed)
    recs, kl_num, kl_den, ent_num = [], 0.0, 0.0, 0.0
    for s in tqdm(range(0, len(rows), args.gen_batch), desc=f"PPO eval [{args.name}]"):
        batch = rows[s:s + args.gen_batch]
        msgs = [prompt_messages(r) for r in batch]
        gen = batch_generate(policy, tok, msgs, int(cfg["max_prompt_length"]), max_new,
                             temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]))
        raw = score_reward_pairs(rm, rm_tok, msgs, gen["responses"], max_length=int(cfg.get("reward_max_length", 1024)))
        pw = gen["prompt_width"]
        for j, r in enumerate(batch):           # per-row stats keep memory low at 768 tokens
            seq, attn = gen["sequences"][j:j + 1], gen["attention_mask"][j:j + 1]
            resp, mask = gen["response_ids"][j:j + 1], gen["response_mask"][j:j + 1]
            lp, ent = policy_token_stats(policy, seq, attn, pw, resp, with_entropy=True)
            with reference_mode(policy):
                rlp, _ = policy_token_stats(policy, seq, attn, pw, resp)
            diff = (lp - rlp) * mask
            kl_num += diff.sum().item()
            kl_den += mask.sum().item()
            ent_num += (ent * mask).sum().item()
            term = gen["terminated_with_eos"][j]
            recs.append({"index": s + j, "prompt_id": r.get("prompt_id"), "response": gen["responses"][j],
                         "length_tokens": gen["response_lengths"][j], "truncated": gen["truncated"][j],
                         "terminated_with_eos": term, "reward_raw": raw[j].item(),
                         "reward_effective": raw[j].item() - (0.0 if term else pen),
                         "kl_seq_sum": diff.sum().item()})

    write_jsonl(res / f"eval_{args.name}_generations.jsonl", recs)
    summary = {
        "name": args.name, "adapter": args.adapter, "num_prompts": len(recs),
        "decoding": {**g, "max_new_tokens": max_new, "seed": seed},
        "reward_raw": summarize([r["reward_raw"] for r in recs]),
        "reward_effective": summarize([r["reward_effective"] for r in recs]),
        "kl_token_mean": kl_num / max(kl_den, 1.0),
        "kl_seq_sum": summarize([r["kl_seq_sum"] for r in recs]),
        "entropy_token_mean": ent_num / max(kl_den, 1.0),
        "length_tokens": summarize([r["length_tokens"] for r in recs]),
        "truncated_rate": float(np.mean([r["truncated"] for r in recs])),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in recs])),
        "wall_clock_sec": timer(),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
    }
    save_json(res / f"eval_{args.name}_summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
