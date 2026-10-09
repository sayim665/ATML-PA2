from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def sequence_logps(model, batch):
    """Summed response-token log-probs under teacher forcing. Returns (logp[B], n_tokens[B])."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    # Batches are LEFT-padded, so derive positions from the mask (otherwise padding shifts RoPE positions).
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    mask = batch["response_mask"][:, 1:]          # token t+1 is predicted at position t
    token_logp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    return (token_logp * mask).sum(-1), mask.sum(-1)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),  # reproducible order
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg, model, loader, optimizer = bundle["cfg"], bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]

    default_out = cfg["standard_output"] if run_name == "standard" else f"outputs/task1_dpo/{run_name}"
    output = repo_path(output_path or default_out)
    output.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg["epochs"])
    max_norm = float(cfg["max_grad_norm"])
    params = trainable_parameters(model)
    device = next(model.parameters()).device

    save_json(results_dir / f"{run_name}_config.json", {
        "run_name": run_name, "dataset": bundle["dataset_path"], "num_examples": len(bundle["rows"]),
        "beta": beta, "learning_rate": float(cfg["learning_rate"]), "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accum, "effective_batch": int(cfg["batch_size"]) * accum, "epochs": epochs,
        "max_sequence_length": int(cfg["max_sequence_length"]), "max_grad_norm": max_norm,
        "seed": int(cfg["seed"]), "lora": cfg["lora"], "dtype": cfg.get("dtype"), "output": str(output),
    })

    timer = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step, micro, skipped, window = 0, 0, 0, []

    def optimizer_step():
        nonlocal step, window
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        rec = {k: sum(w[k] for w in window) / len(window) for k in window[0]}
        rec.update({"step": step, "grad_norm": float(grad_norm), "elapsed_sec": timer()})
        append_jsonl(log_path, rec)
        if step == 1 or step % 10 == 0:
            print(f"step {step:4d} | loss {rec['loss']:.4f} | acc {rec['pref_acc']:.3f} | "
                  f"margin {rec['reward_margin']:+.3f} | gnorm {rec['grad_norm']:.3f} | {rec['elapsed_sec']/60:.1f} min")
        window = []

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        for chosen, rejected in tqdm(loader, desc=f"DPO[{run_name}] epoch {epoch + 1}/{epochs}"):
            chosen, rejected = to_device(chosen, device), to_device(rejected, device)

            # Reference = same base weights with the LoRA adapter switched off.
            with torch.no_grad(), reference_mode(model):
                ref_c, _ = sequence_logps(model, chosen)
                ref_r, _ = sequence_logps(model, rejected)

            pol_c, len_c = sequence_logps(model, chosen)
            pol_r, len_r = sequence_logps(model, rejected)
            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)

            if not torch.isfinite(loss):
                skipped += 1
                print(f"WARNING: non-finite loss at micro-batch {micro}; skipped")
                continue

            (loss / accum).backward()
            micro += 1
            chosen_reward = (beta * (pol_c - ref_c)).detach()
            rejected_reward = (beta * (pol_r - ref_r)).detach()
            window.append({
                "loss": loss.item(),
                "pref_acc": diag["preference_accuracy"].item(),
                "logit_mean": diag["logit_mean"].item(),
                "chosen_reward": chosen_reward.mean().item(),
                "rejected_reward": rejected_reward.mean().item(),
                "reward_margin": (chosen_reward - rejected_reward).mean().item(),
                "chosen_len": len_c.float().mean().item(),
                "rejected_len": len_r.float().mean().item(),
            })
            if micro % accum == 0:
                optimizer_step()

    if window:  # flush a final partial accumulation window
        optimizer_step()

    model.save_pretrained(str(output))
    summary = {
        "run_name": run_name, "optimizer_steps": step, "micro_batches": micro,
        "skipped_nonfinite": skipped, "wall_clock_sec": timer(),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "adapter": str(output),
    }
    save_json(results_dir / f"{run_name}_train_summary.json", summary)
    print(summary)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
