#!/usr/bin/env bash
set -euo pipefail

# 1) Define an array of MODEL_CONFIGs
MODEL_CONFIGS=(
    "model_configs/Qwen3-06B.yaml"
    "model_configs/Qwen3-06B-Base.yaml"
    "model_configs/MNLP_M2_mcqa_model.yaml"
    "model_configs/MNLP_M2_mcqa_model_cot00.yaml"
    "model_configs/MNLP_M2_mcqa_model_cot02.yaml"
    "model_configs/MNLP_M2_mcqa_model_cot05.yaml"
    "model_configs/MNLP_M2_mcqa_model_cot08.yaml"
    "model_configs/MNLP_M2_mcqa_model_cot10.yaml"
)

# 2) Common TASKS and OUTPUT_DIR
TASKS1="community|mnlp_mcqa_evals|0|0,community|mmlu:stem|0|0,helm|med_qa|0|0,helm|commonsenseqa|0|0,original|arc:c:letters|0|0"
TASKS2="lighteval|agieval:aqua-rat|0|0,lighteval|sciq|0|0,lighteval|openbookqa|0|0,lighteval|race:high|0|0"
TASKS3="original|mmlu|0|0"
OUTPUT_DIR="outputs"

# 3) Loop through each config and run both evals
for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
  echo "=== Running with $MODEL_CONFIG ==="
  # lighteval accelerate \
  #   --eval-mode lighteval \
  #   --save-details \
  #   --custom-tasks "community_tasks/mnlp_mcqa_evals.py" \
  #   --output-dir "${OUTPUT_DIR}" \
  #   "${MODEL_CONFIG}" \
  #   "${TASKS1}"

  # lighteval accelerate \
  #   --eval-mode lighteval \
  #   --save-details \
  #   --output-dir "${OUTPUT_DIR}" \
  #   "${MODEL_CONFIG}" \
  #   "${TASKS2}"

  lighteval accelerate \
    --eval-mode lighteval \
    --save-details \
    --output-dir "${OUTPUT_DIR}" \
    "${MODEL_CONFIG}" \
    "${TASKS3}"
    --override-batch-size 16
done







# ### ←–– EDIT THESE
# MODEL_CONFIG="model_configs/Qwen3-06B.yaml"
# #MODEL_CONFIG="model_configs/Qwen3-06B-Base.yaml"
# #MODEL_CONFIG="model_configs/MNLP_M2_mcqa_model_cot00.yaml"
# #MODEL_CONFIG="model_configs/MNLP_M2_mcqa_model_cot02.yaml"
# #MODEL_CONFIG="model_configs/MNLP_M2_mcqa_model_cot05.yaml"
# #MODEL_CONFIG="model_configs/MNLP_M2_mcqa_model_cot08.yaml"
# #MODEL_CONFIG="model_configs/MNLP_M2_mcqa_model_cot10.yaml"

# TASKS1="community|mnlp_mcqa_evals|0|0,community|mmlu:stem|0|0,helm|med_qa|0|0,helm|commonsenseqa|0|0,original|arc:c:letters|0|0"
# TASKS2="lighteval|agieval:aqua-rat|0|0,lighteval|sciq|0|0,lighteval|openbookqa|0|0,lighteval|race:high|0|0"
# OUTPUT_DIR="outputs"
# ### –––––––––––––––––––––

# lighteval accelerate \
#   --eval-mode lighteval \
#   --save-details \
#   --custom-tasks "community_tasks/mnlp_mcqa_evals.py" \
#   --output-dir "${OUTPUT_DIR}" \
#   "${MODEL_CONFIG}" \
#   "${TASKS1}"

#   lighteval accelerate \
#   --eval-mode lighteval \
#   --save-details \
#   --output-dir "${OUTPUT_DIR}" \
#   "${MODEL_CONFIG}" \
#   "${TASKS2}"




# # Evaluating MCQA Model
# # lighteval accelerate \
# #     --eval-mode "lighteval" \
# #     --save-details \
# #     --custom-tasks "community_tasks/mnlp_mcqa_evals.py" \
# #     --output-dir "outputs" \
# #     model_configs/mcqa_model.yaml \
# #     "community|mnlp_mcqa_evals|0|0"

# # lighteval accelerate \
# #   --eval-mode lighteval \
# #   --save-details \
# #   --output-dir outputs \
# #   model_configs/mcqa_model.yaml \
# #   "helm|med_qa|0|0,lighteval|agieval:aqua-rat|0|0,lighteval|sciq|0|0,original|arc:c:letters|0|0,lighteval|openbookqa|0|0,lighteval|race:high|0|0,helm|commonsenseqa|0|0"

  
  
  
