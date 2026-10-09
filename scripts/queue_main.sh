#!/usr/bin/env bash
cd /content/drive/MyDrive/ATML-PA2-LLM-PostTraining
run() { echo ">>> $(date +%H:%M) $*"; "$@" || echo "FAILED: $*"; }

# ---- Task 2: PPO standard (needed by Task 4) ----
if python -u -m task2_ppo.continue_train --config configs/ppo.yaml --run-name smoke --updates 1; then
  run python -u -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
else echo "FAILED: PPO smoke test - skipping PPO standard"; fi
run python -u -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
run python -u -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard

# ---- Task 3: GRPO standard (needed by Task 4) ----
if python -u -m task3_grpo.continue_train --config configs/grpo.yaml --run-name smoke --updates 1; then
  run python -u -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
else echo "FAILED: GRPO smoke test - skipping GRPO standard"; fi
run python -u -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
run python -u -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard

# ---- Task 2 ablations, then Task 3 normalization ----
run python -u -m task2_ppo.analyze_clipping --config configs/ppo.yaml
run python -u -m task2_ppo.ablate_kl --config configs/ppo.yaml
run python -u -m task3_grpo.compare_normalization --config configs/grpo.yaml

# ---- Task 4 ----
run python -u -m task4_safety.generate_responses --config configs/feedback.yaml
run python -u -m task4_safety.make_audit_sheet --config configs/feedback.yaml
run python -u -m task4_safety.judge_responses --config configs/feedback.yaml
run python -u -m task4_safety.evaluate_safety --config configs/feedback.yaml
echo ">>> $(date +%H:%M) MAIN QUEUE DONE"
