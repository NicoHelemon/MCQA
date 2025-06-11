#!/usr/bin/env bash

# train_mcqa.sh
# Usage: bash train_mcqa.sh

# ─── User-specific settings ───────────────────────────────────────────────────
# If you leave HF_TOKEN and HF_USERNAME as empty strings ("", ""), models will only be saved locally
# under: output_dir = f"tmp/mcqa_model{my_model_suffix}".
# If you provide valid values, the models will also be uploaded to Hugging Face under:
# {HF_USERNAME}/{MY_MODEL_NAME}_{MY_MODEL_SUFFIX}_e{i} for each epoch i.
# Example: NicoHelemon/MNLP_M3_mcqa_model_TA_cot00_e1

# HF_TOKEN="Your_HF_Token"
# HF_USERNAME="Your_HF_Username"
#HF_TOKEN=""
#HF_USERNAME=""
HF_TOKEN="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"
HF_USERNAME="NicoHelemon"

# ─── Fixed training settings ─────────────────────────────────────────────────$
MY_MODEL_NAME="MNLP_M3_mcqa_model_for_TA"
MY_MODEL_SUFFIX="cot00"
PROMPT_TYPE="cot"
COT_PROB=0
NUM_EPOCHS=1
BATCH_SIZE=64
GRAD_ACC_STEPS=4
DATA_SUBSET=0

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
