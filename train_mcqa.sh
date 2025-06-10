#!/usr/bin/env bash

# run_train.sh
# Usage: bash run_train.sh

# ─── User-specific settings ───────────────────────────────────────────────────
HF_TOKEN="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"
HF_USERNAME="NicoHelemon"

# ─── Fixed training settings ─────────────────────────────────────────────────$
MY_MODEL_NAME="MNLP_M3_mcqa_model_for_TA"
MY_MODEL_SUFFIX="cot00"
PROMPT_TYPE="legacy"
COT_PROB=0
NUM_EPOCHS=3
BATCH_SIZE=64
GRAD_ACC_STEPS=4
DATA_SUBSET=0

# PLEASE READ
# The trained models will be uploaded to Hugging Face under the following names:
#   for i in range(1, NUM_EPOCHS + 1):
#     {HF_USERNAME}/{MY_MODEL_NAME}_{MY_MODEL_SUFFIX}_e{i}
# e.g., NicoHelemon/MNLP_M3_mcqa_model_TA_cot00_e1, NicoHelemon/MNLP_M3_mcqa_model_TA_cot00_e2, ...

# ─── Invoke the training script ──────────────────────────────────────────────
python models/train.py \
  --hf_token        "${HF_TOKEN}" \
  --hf_username     "${HF_USERNAME}" \
  --my_model_name   "${MY_MODEL_NAME}" \
  --my_model_suffix "${MY_MODEL_SUFFIX}" \
  --prompt_type     "${PROMPT_TYPE}" \
  --cot_prob        "${COT_PROB}" \
  --num_epochs      "${NUM_EPOCHS}" \
  --batch_size      "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
  --data_subset     "${DATA_SUBSET}"
