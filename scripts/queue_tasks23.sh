#!/usr/bin/env bash
cd /content/drive/MyDrive/ATML-PA2-LLM-PostTraining
run() { echo ">>> $(date +%H:%M) $*"; "$@" || echo "FAILED: $*"; }

# wait for Task 1 jobs and the queued PPO standard chain
while pgrep -f "[t]ask1_dpo|[t]ask2_ppo" > /dev/null; do sleep 30; done

# 1) GRPO standard first: needed by Task 4
run python -u -m task3_grpo.continue_train --config configs/grpo.yaml --run-name smoke --updates 1
run python -u -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
run python -u -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
run python -u -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard

# 2) PPO ablations (cached clipping batch + 5 matched forks, the shared one runs once)
run python -u -m task2_ppo.analyze_clipping --config configs/ppo.yaml
run python -u -m task2_ppo.ablate_kl --config configs/ppo.yaml

# 3) GRPO normalization forks
run python -u -m task3_grpo.compare_normalization --config configs/grpo.yaml
echo ">>> $(date +%H:%M) QUEUE DONE"
