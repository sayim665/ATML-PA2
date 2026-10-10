# ATML PA2 — LLM Post-Training (Sayim, `sayim665`)

Submission repository for EE-5102 / CS-6304 Programming Assignment 2. It compares DPO, PPO and GRPO under
controlled interventions, evaluates safety calibration on XSTest, and contrasts verifiable rewards (RLVR) with
AI feedback (RLAIF). Everything runs from the repository root as `python -m ...` scripts; no notebook state is
needed to reproduce any reported number.

Built on the course starter: <https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining>
(course assets pinned at revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`).

## Environment

All experiments ran on Google Colab with a single **Tesla T4 (15 GB)**, Python 3.13, PyTorch 2.11 (CUDA),
Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, Tokenizers 0.22.1. Global seed **6304** (from `configs/base.yaml`).

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
```

Downloaded checkpoints, cached rollouts, course data and trained adapters (`outputs/`) are git-ignored. Every
JSON/JSONL/CSV result cited in the report is committed under `results/`.

## The three objective defects and how they were validated

Each of Tasks 1–3 shipped with one deliberate defect in its core objective. All three are fixed, and
`python -m scripts.validate_objectives` reproduces the evidence (it evaluates the corrected objective and,
for comparison, re-implements the original defective formula).

| Task | File | Defect | Fix | Evidence |
|---|---|---|---|---|
| 1 DPO | `task1_dpo/dpo.py` | logits used `policy_margin + ref_margin` | `policy_margin - ref_margin` | loss at policy = reference is ln 2 = 0.6931 (defect gave 0.0181) |
| 2 PPO | `task2_ppo/ppo.py` | clipped surrogate took `torch.maximum` | `torch.minimum` | objective −0.6 and gradients only on unclipped tokens (defect: +0.6, inverted gradient pattern) |
| 3 GRPO | `task3_grpo/grpo.py` | advantages normalised over the whole batch, ignoring `group_ids` | per-prompt mean/std | advantages `[-1, 1, -1, 1]` (defect leaked prompt difficulty: `[0.78, 1.18, -1.18, -0.78]`) |

The GRPO defect is invisible in the standard continuation (`prompts_per_update: 1` makes every batch one group)
but corrupts any multi-group batch, such as the group-size regrouping.

## Reproducing every experiment

Commands are listed in the order they were run. `results/<task>/` receives logs, configs, summaries and every
generated response.

### Task 1 — DPO
```bash
python -m task1_dpo.train    --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft      # untouched SFT baseline
python -m task1_dpo.ablate_beta    --config configs/dpo.yaml   # beta in {0.03, 0.10, 0.30}, first 600 valid pairs each
python -m task1_dpo.analyze_length --config configs/dpo.yaml   # length-balanced model + stratified evaluation
```

### Task 2 — PPO
```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard     # 20 updates from the midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml   # cached batch + eps in {0.05, 0.20, 0.50} forks
python -m task2_ppo.ablate_kl        --config configs/ppo.yaml   # beta_KL in {0, 0.10, 0.20} forks
```
The ε = 0.20 / βKL = 0.10 fork is the baseline setting of both studies and is trained once and shared
(`task2_ppo/forks.py`). Every fork starts from the identical supplied policy and critic, uses the same prompt
order and per-update sampling seed, and runs the same 8-update budget.

### Task 3 — GRPO
```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard   # 20 updates, K = 4
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size     --config configs/grpo.yaml   # CPU only: K in {2, 4, 8} on the supplied cache
python -m task3_grpo.compare_normalization  --config configs/grpo.yaml   # canonical vs Dr. GRPO, 8-update forks
```
Group-size regrouping: each prompt's 8 cached completions (in `generation_index` order) are split into 8/K
disjoint groups, so every K uses the same 192 completions. Difficulty bins are tertiles of each prompt's mean
reward over its 8 completions.

### Task 4 — Safety calibration
```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml   # SFT/DPO/PPO/GRPO, greedy, 450 XSTest prompts
python -m task4_safety.make_audit_sheet   --config configs/feedback.yaml   # fixed 60 prompts -> blind, de-duplicated sheet
python -m task4_safety.label_audit                                         # interactive manual labelling (done before viewing judge labels)
python -m task4_safety.judge_responses    --config configs/feedback.yaml   # supplied judge, cached and resumable
python -m task4_safety.evaluate_safety    --config configs/feedback.yaml   # rates, category table, audit agreement
```
The audit sheet joins the 60 fixed prompts (30 safe, 30 unsafe, from the starter's seeded sampler) to all four
policies' responses, merges identical responses (110 distinct), shuffles them, and hides both the policy name and
the judge label.

### Task 5 — RLVR vs RLAIF
```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm        # 300 GSM8K problems
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer   # fixed 100-problem SVAMP set
python -m task5_feedback.score_perturbations --config configs/feedback.yaml               # 20 x 5 controlled diagnostics
python -m task5_feedback.compare_feedback    --config configs/feedback.yaml
```

### Secondary analyses cited in the report
```bash
python -m scripts.validate_objectives   # the three objective fixes (above)
python -m scripts.extra_analyses        # DPO-vs-SFT paired test, likelihood displacement, prompt-length check, PPO identical-response count
```

`scripts/queue_main.sh` is the convenience runner used on Colab to chain the Task 2–4 jobs. It uses an
absolute Colab Drive path; adjust it before using elsewhere.

## Implementation choices (beyond the starter)

- **DPO long-prompt filter.** Pairs whose prompt leaves fewer than 64 response tokens inside the
  768-token sequence cap are dropped before training and evaluation (57/1,500 training pairs, 11/300 held-out
  pairs). Every dropped index is logged in `results/task1_dpo/filtered_*.json`. The filter runs before the
  `--max-examples` cut, so the β forks share the same first 600 valid pairs.
- **Correct positions under left padding.** Log-probabilities are computed with `position_ids` derived from the
  attention mask.
- **No dropout inside PPO/GRPO updates.** Policies are kept in eval mode (PPO) or in train mode with all dropout
  modules disabled (GRPO), so the importance ratio is exactly 1 before the first step and the clip fraction is not
  inflated by dropout noise.
- **PPO critic head trained in fp32** under autocast, avoiding fp16 Adam underflow.
- **Inference-mode tensors.** Generated sequences from the course `batch_generate` helper are cloned before
  entering any autograd computation.
- **Evaluation protocols (fixed across every compared condition).** DPO: 128 held-out prompts, sampled decoding
  (T = 0.7, top-p = 0.9), 256-token cap, seed 6304; the KL estimator is token-averaged over sampled responses.
  PPO/GRPO: the first 32 held-out prompts at a 768-token cap, the protocol recorded in the course's
  `ppo_midpoint_acceptance.json`. Tasks 4–5: greedy decoding.
- **Task 5 pairwise ties.** When a policy's answer is word-for-word identical to SFT's, the comparison is recorded
  as a tie without calling the judge; the count is reported as `identical_to_sft`.

## Known limitations (all from the fixed course configuration; deliberately left unchanged)

- **Prompt truncation before generation.** `max_prompt_length: 256` truncates from the right, removing the
  assistant cue for 21.3% of RL training prompts (3/32 held-out prompts). Those prompts produce document
  continuations rather than answers. 1.2% of training prompts also exceed the GRPO reward window (1,024 tokens;
  0.8% exceed PPO's 1,280), so the reward model never sees the response. See `results/pipeline_prompt_length_check.json`.
- **Weak supplied critic.** Held-out explained variance −3.67 per the course release; our continuation logs show
  negative explained variance throughout.
- **Unequal RLVR/RLAIF budgets.** The supplied adapters were trained for 80 and 50 steps respectively (same K and
  similar lengths), so RLVR saw roughly 1.6x more generated tokens.
- **SVAMP prompt wording differs** from the GSM8K training instruction, so part of any transfer gap may be
  instruction shift rather than distribution shift.
- **Small held-out sets** (32 prompts for PPO/GRPO) and the sampled KL estimator mean that differences below about
  ±0.25 reward or ±2×10⁻⁴ KL per token are within noise.

## Where each result lives

| Task | Key files in `results/` |
|---|---|
| 1 | `task1_dpo/eval_{standard,sft}_summary.json`, `beta_sweep.csv`, `length_study.json`, `length_dataset_profile.json`, `likelihood_displacement.json`, `standard_vs_sft_paired.json` |
| 2 | `task2_ppo/standard_train_log.jsonl`, `eval_{midpoint,standard}_summary.json`, `clipping_cached_batch.json`, `clipping_forks.csv`, `kl_forks.json` |
| 3 | `task3_grpo/standard_train_log.jsonl`, `eval_{midpoint,standard}_summary.json`, `group_size_study.json`, `normalization_comparison.json`, `uninformative_group_example.json` |
| 4 | `task4_safety/safety_summary.json`, `safety_policy_table.csv`, `category_label_counts.csv`, `manual_audit_sheet.csv`, `audit_disagreements.csv` |
| 5 | `task5_feedback/eval_{gsm,transfer}_summary.json`, `diagnostics_summary.json`, `task5_comparison.json` |

Peak VRAM and wall-clock time for every training run are in the corresponding `*_train_summary.json`.

## Attribution

- **Course starter code** (course staff): everything in `common/`, `scripts/download_assets.py`,
  `scripts/validate_assets.py`, `scripts/check_environment.py`, the configuration files, the task scaffolds, and
  the supplied components used unchanged: the Task 4 judge prompt, loader and parser
  (`task4_safety/judge_responses.py`), and the Task 5 verifier and pairwise judge (`task5_feedback/rlvr.py`,
  `task5_feedback/rlaif.py`).
- **Written for this assignment:** the three objective fixes; the DPO, PPO and GRPO training and continuation
  loops; all evaluation, ablation and analysis scripts; `task2_ppo/forks.py`; `task4_safety/label_audit.py`;
  `scripts/validate_objectives.py`; `scripts/extra_analyses.py`; `scripts/queue_main.sh`.
- **Libraries:** PyTorch, Hugging Face Transformers, PEFT, bitsandbytes, pandas, NumPy, scikit-learn.
- **Models and data** (public, downloaded at runtime): Qwen2.5-0.5B/1.5B/3B-Instruct,
  `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`, UltraFeedback, XSTest, GSM8K, SVAMP.
