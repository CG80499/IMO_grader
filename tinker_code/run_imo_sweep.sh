#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs/imo_runs"
mkdir -p "${LOG_DIR}"

models=(
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-32B"
)

learning_rates=(
  "5e-5"
  # "2e-5"
  # "1e-5"
)

num_epochs_per_step=(
  "1"
)

temperatures=(
  "0.7"
)

use_high_precision_reward_fns=(
  "false"
  "true"
)

for model in "${models[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for epochs in "${num_epochs_per_step[@]}"; do
      for temp in "${temperatures[@]}"; do
        for use_high_precision_reward_fn in "${use_high_precision_reward_fns[@]}"; do
          model_slug=$(echo "${model}" | tr '/:' '__')
          log_file="${LOG_DIR}/imo_${model_slug}_lr${lr}_epochs${epochs}_temp${temp}_use_high_precision_reward_fn${use_high_precision_reward_fn}.log"

          echo "Launching ${model} | lr=${lr} | epochs=${epochs} | temp=${temp} | use_high_precision_reward_fn=${use_high_precision_reward_fn}"
          nohup uv run imo_verifer_training.py \
            --model-name "${model}" \
            --learning-rate "${lr}" \
            --num-epochs-per-step "${epochs}" \
            --temperature "${temp}" \
            --use-high-precision-reward-fn "${use_high_precision_reward_fn}" \
            > "${log_file}" 2>&1 &
        done
      done
    done
  done
done

echo "All IMO jobs launched. Logs in ${LOG_DIR}."