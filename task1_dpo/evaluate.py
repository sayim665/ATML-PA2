from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import word_count, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.train import filter_long_prompts, make_collate, sequence_logps, to_device

STRATA_KEYS = ("stratum", "length_stratum", "length_bucket", "bucket", "strata")


def summarize(xs):
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    q1, q3 = np.percentile(a, [25, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)),
            "iqr": float(q3 - q1), "n": int(a.size)}


@torch.no_grad()
def preference_eval(model, tokenizer, rows, cfg, beta, strata_key=None, batch_size=2):
    """Held-out DPO margin m_theta, loss and accuracy for every pair (reference = adapter disabled)."""
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(model.parameters()).device
    out = []
    for start in tqdm(range(0, len(rows), batch_size), desc="preference eval"):
        chunk = rows[start:start + batch_size]
        chosen, rejected = collate(chunk)
        chosen, rejected = to_device(chosen, device), to_device(rejected, device)
        pc, lc = sequence_logps(model, chosen)
        pr, lr = sequence_logps(model, rejected)
        with reference_mode(model):
            rc, _ = sequence_logps(model, chosen)
            rr, _ = sequence_logps(model, rejected)
        margin = (pc - rc) - (pr - rr)
        loss = -F.logsigmoid(beta * margin)
        for j, row in enumerate(chunk):
            out.append({
                "index": row["_idx"], "id": row.get("prompt_id", row.get("id")),
                "stratum": row.get(strata_key) if strata_key else None,
                "margin": margin[j].item(), "loss": loss[j].item(), "correct": bool(margin[j].item() > 0),
                "chosen_logratio": (pc[j] - rc[j]).item(), "rejected_logratio": (pr[j] - rr[j]).item(),
                "chosen_tokens": int(lc[j].item()), "rejected_tokens": int(lr[j].item()),
            })
    return out


@torch.no_grad()
def generate_and_score(model, tokenizer, prompts, cfg, max_new, batch_size, rm=None, rm_tok=None):
    """Sample one response per prompt; record reward, length and sampled KL vs reference."""
    g = cfg["generation"]
    max_prompt = int(cfg["max_sequence_length"])
    records, kl_num, kl_den = [], 0.0, 0.0
    for start in tqdm(range(0, len(prompts), batch_size), desc="generate"):
        batch = prompts[start:start + batch_size]
        msgs = [p["messages"] for p in batch]
        gen = batch_generate(model, tokenizer, msgs, max_prompt, max_new,
                             temperature=float(g["temperature"]), top_p=float(g["top_p"]),
                             do_sample=bool(g["do_sample"]))
        pol_lp, logits = response_token_logprobs(model, gen["sequences"], gen["attention_mask"],
                                                 gen["prompt_width"], gen["response_ids"])
        del logits
        with reference_mode(model):
            ref_lp, logits = response_token_logprobs(model, gen["sequences"], gen["attention_mask"],
                                                     gen["prompt_width"], gen["response_ids"])
            del logits
        mask = gen["response_mask"]
        diff = (pol_lp - ref_lp) * mask
        kl_num += diff.sum().item()
        kl_den += mask.sum().item()
        rewards = score_reward_pairs(rm, rm_tok, msgs, gen["responses"]) if rm is not None else None
        for j, p in enumerate(batch):
            records.append({
                **{k: v for k, v in p.items() if k != "messages"},
                "response": gen["responses"][j],
                "length_tokens": gen["response_lengths"][j],
                "truncated": gen["truncated"][j],
                "terminated_with_eos": gen["terminated_with_eos"][j],
                "kl_seq_sum": diff[j].sum().item(),
                "reward": rewards[j].item() if rewards is not None else None,
            })
    # Token-level aggregate = masked mean over all response tokens (same convention as common.metrics.sampled_kl).
    kl_token_mean = kl_num / max(kl_den, 1.0)
    return records, kl_token_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the untouched SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta for held-out DPO loss (default: config beta)")
    ap.add_argument("--pairs", help="held-out pair file (default: dpo_standard_eval)")
    ap.add_argument("--num-gen", type=int, default=128, help="held-out prompts used for generation metrics")
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-word-limit", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    beta = float(args.beta if args.beta is not None else cfg["beta"])
    max_new = int(cfg["max_generation_tokens"])
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    timer = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer = load_tokenizer(cfg["base_model"])
    adapter = None if args.adapter.lower() == "none" else args.adapter
    model = load_policy(cfg, adapter_path=adapter, trainable=False)

    # ---------- held-out preference metrics ----------
    pairs_path = args.pairs or cfg["paths"]["dpo_standard_eval"]
    rows = [dict(r, _idx=i) for i, r in enumerate(read_jsonl(pairs_path))]
    rows, dropped = filter_long_prompts(rows, tokenizer, int(cfg["max_sequence_length"]))
    strata_key = next((k for k in STRATA_KEYS if k in rows[0]), None)
    if strata_key is None and "length" in str(pairs_path):
        print(f"WARNING: no stratum field found. Row keys: {list(rows[0].keys())}")

    pairs = preference_eval(model, tokenizer, rows, cfg, beta, strata_key)
    write_jsonl(results_dir / f"eval_{args.name}_pairs.jsonl", pairs)
    summary = {
        "name": args.name, "adapter": args.adapter, "pairs_file": str(pairs_path), "beta_for_loss": beta,
        "pairs_evaluated": len(pairs), "pairs_dropped_long_prompt": len(dropped),
        "dropped_indices": [d["index"] for d in dropped],
        "heldout_dpo_loss": float(np.mean([p["loss"] for p in pairs])),
        "heldout_pref_accuracy": float(np.mean([p["correct"] for p in pairs])),
        "heldout_margin": summarize([p["margin"] for p in pairs]),
    }
    if strata_key:
        summary["strata_key"] = strata_key
        summary["by_stratum"] = {
            s: {"n": len(g), "pref_accuracy": float(np.mean([p["correct"] for p in g])),
                "dpo_loss": float(np.mean([p["loss"] for p in g])),
                "margin_mean": float(np.mean([p["margin"] for p in g]))}
            for s in sorted({p["stratum"] for p in pairs})
            for g in [[p for p in pairs if p["stratum"] == s]]
        }

    # ---------- generation metrics: reward, KL, length ----------
    if not args.skip_generation:
        rm, rm_tok = load_reward_model(cfg)
        gen_prompts = [{"index": r["_idx"], "id": r.get("prompt_id", r.get("id")),
                        "messages": prompt_messages_from_preference(r)} for r in rows[: args.num_gen]]
        set_seed(seed)
        gens, kl_tok = generate_and_score(model, tokenizer, gen_prompts, cfg, max_new, args.gen_batch, rm, rm_tok)
        write_jsonl(results_dir / f"eval_{args.name}_generations.jsonl", gens)
        summary.update({
            "gen_prompts": len(gens),
            "decoding": {**cfg["generation"], "max_new_tokens": max_new, "seed": seed},
            "kl_token_mean": kl_tok,
            "kl_seq_sum": summarize([g["kl_seq_sum"] for g in gens]),
            "reward": summarize([g["reward"] for g in gens]),
            "length_tokens": summarize([g["length_tokens"] for g in gens]),
            "truncated_rate": float(np.mean([g["truncated"] for g in gens])),
        })

    # ---------- word-limit compliance on the common prompt set ----------
    if not args.skip_word_limit:
        wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
        wl_prompts = []
        for i, r in enumerate(wl_rows):
            msgs = prompt_messages(r)
            user_text = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
            wl_prompts.append({"index": i, "id": r.get("prompt_id", r.get("id")), "prompt_text": user_text, "messages": msgs})
        set_seed(seed)
        wl, _ = generate_and_score(model, tokenizer, wl_prompts, cfg, max_new, args.gen_batch)
        for rec in wl:
            rec["word_count"] = word_count(rec["response"])
            rec["compliant"] = word_limit_compliance(rec["prompt_text"], rec["response"])
        write_jsonl(results_dir / f"eval_{args.name}_wordlimit.jsonl", wl)
        scored = [r["compliant"] for r in wl if r["compliant"] is not None]
        summary.update({
            "word_limit_prompts": len(wl), "word_limit_parsed": len(scored),
            "word_limit_compliance": float(np.mean(scored)) if scored else None,
            "word_limit_length_tokens": summarize([r["length_tokens"] for r in wl]),
            "word_limit_word_count": summarize([r["word_count"] for r in wl]),
        })

    summary["wall_clock_sec"] = timer()
    summary["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    save_json(results_dir / f"eval_{args.name}_summary.json", summary)
    print({k: v for k, v in summary.items() if k != "dropped_indices"})


if __name__ == "__main__":
    main()
