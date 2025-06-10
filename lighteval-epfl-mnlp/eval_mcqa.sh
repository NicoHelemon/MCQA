#!/usr/bin/env bash
set -euo pipefail

# 1) Define an array of MODEL_CONFIGs
MODEL_CONFIGS=(
      "model_configs_mcqa/Qwen3-06B.yaml"
      "model_configs_mcqa/Qwen3-06B-Base.yaml"
      "model_configs_mcqa/MNLP_M2_mcqa_model_cot00_e1.yaml"
      "model_configs_mcqa/MNLP_M2_mcqa_model_cot00_e3.yaml"
      "model_configs_mcqa/MNLP_M2_mcqa_model_cot10_e1.yaml"
      "model_configs_mcqa/MNLP_M2_mcqa_model_cot10_e3.yaml"
      "model_configs_mcqa/MNLP_M3_mcqa_model_cot00_e1.yaml"
      "model_configs_mcqa/MNLP_M3_mcqa_model_cot00_e3.yaml"
      "model_configs_mcqa/MNLP_M3_mcqa_model_cot10_e1.yaml"
      "model_configs_mcqa/MNLP_M3_mcqa_model_cot10_e3.yaml"
)

# 2) Common TASKS and OUTPUT_DIR
TASKS1="community|mnlp_mcqa_evals_legacy|0|0,community|mnlp_mcqa_evals|0|0,community|MNLP_M3_mcqa_dataset|0|0,community|mmlu:stem|0|0,helm|med_qa|0|0,helm|commonsenseqa|0|0,original|arc:c:letters|0|0"
TASKS2="lighteval|agieval:aqua-rat|0|0,lighteval|sciq|0|0,lighteval|openbookqa|0|0,lighteval|race:high|0|0"
TASKS3="original|mmlu|0|0"


TASKS4="community|mnlp_mcqa_evals_legacy|0|0,community|mnlp_mcqa_evals|0|0"
TASKS5="community|mmlu:stem|0|0"
TASKS6="community|mnlp_mcqa_evals|0|0"
OUTPUT_DIR="outputs_mcqa"
EVAL_MODE="lighteval"

# 3) Loop through each config and run both evals
for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
  echo "=== Running with $MODEL_CONFIG ==="
  # lighteval accelerate \
  #   --eval-mode "${EVAL_MODE}" \
  #   --save-details \
  #   --custom-tasks "community_tasks/my_evals.py" \
  #   --output-dir "${OUTPUT_DIR}" \
  #   "${MODEL_CONFIG}" \
  #   "${TASKS1}"

  # lighteval accelerate \
  #   --eval-mode "${EVAL_MODE}" \
  #   --save-details \
  #   --output-dir "${OUTPUT_DIR}" \
  #   "${MODEL_CONFIG}" \
  #   "${TASKS2}"

  # lighteval accelerate \
  #   --eval-mode "${EVAL_MODE}" \
  #   --save-details \
  #   --output-dir "${OUTPUT_DIR}" \
  #   "${MODEL_CONFIG}" \
  #   "${TASKS3}"

    lighteval accelerate \
    --eval-mode "${EVAL_MODE}" \
    --save-details \
    --custom-tasks "community_tasks/my_evals.py" \
    --output-dir "${OUTPUT_DIR}" \
    "${MODEL_CONFIG}" \
    "${TASKS6}"
    
done
