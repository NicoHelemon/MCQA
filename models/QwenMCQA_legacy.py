# train_mcqa_model.py

from unsloth import FastModel
import os
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import PeftModel
from huggingface_hub import HfApi

def count_params(model, label="Model"):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{label} nb params:   total = {total:,}   trainable = {trainable:,}")

def fine_tune_peft_model(
    preprocess,
    MY_MODEL_NAME  : str = "",
    NUM_EPOCHS     : int = 1,
    SAVE_STEPS     : int = 50,
    LORA_R         : int = 128,
    LORA_ALPHA     : int = 256,
    DATA_SUBSET    : int = 0,
    commit_message : str = "new commit"):

    if MY_MODEL_NAME != "" and not MY_MODEL_NAME.startswith("_"):
        MY_MODEL_NAME = "_" + MY_MODEL_NAME

    # ─── Configuration ────────────────────────────────────────────────────────────
    HF_TOKEN     = "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"
    os.environ["HF_TOKEN"] = HF_TOKEN
    HF_USERNAME  = "NicoHelemon"
    REPO_DS      = f"{HF_USERNAME}/MNLP_M2_mcqa_dataset"
    HUB_MODEL_ID = f"{HF_USERNAME}/MNLP_M2_mcqa_model{MY_MODEL_NAME}"
    MODEL_NAME   = "Qwen/Qwen3-0.6B-Base"
    OUTPUT_DIR   = f"tmp/mcqa_model{MY_MODEL_NAME}"
    MAX_LENGTH   = 512
    BATCH_SIZE   = 512
    NUM_EPOCHS   = NUM_EPOCHS
    LEARNING_RATE = 3e-5
    LOGGING_STEPS = 10
    SAVE_STEPS    = SAVE_STEPS
    GRADIENT_ACCUMULATION_STEPS = 1
    LORA_R        = LORA_R
    LORA_ALPHA    = LORA_ALPHA
    
    # ─── Dataset Loading ─────────────────────────────────────────────────────────
    raw = load_dataset(REPO_DS)
    if DATA_SUBSET == 0:
        train_raw = raw["train"].shuffle(seed=42)
    else:
        train_raw = raw["train"].shuffle(seed=42).select(range(DATA_SUBSET))
    # ─── Model & Tokenizer Setup via Unsloth ─────────────────────────────────────
    model, tokenizer = FastModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_LENGTH,
        token          = HF_TOKEN,
    )
    count_params(model, "Base")
    model.gradient_checkpointing_enable()
    
    # ─── Inject LoRA adapters via Unsloth ─────────────────────────────────────────
    model = FastModel.get_peft_model(
        model,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = 0.05,
        target_modules = [
            "q_proj","k_proj","v_proj","o_proj",
            "up_proj","down_proj",
            "layer_norm1","layer_norm2"
        ],
        bias           = "all",
    )

    count_params(model, "PEFT-wrapped")
    
    # Configure tokenizer for padding/truncation
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_marker = tokenizer("Answer:").input_ids

    train_ds = train_raw.map(
        preprocess,
        fn_kwargs={"tokenizer": tokenizer, 
                   "MAX_LENGTH": MAX_LENGTH,
                   "tokenized_marker": tokenized_marker},
        remove_columns=train_raw.column_names,
    )
    
    # ─── Training Arguments and Trainer ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_token=HF_TOKEN,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim="adamw_torch",
        bf16=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_dir=f"{OUTPUT_DIR}/logs",            # where TB will look
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        report_to=["tensorboard"],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )

    # ─── Training ──────────────────────────────────────────
    
    print("Starting training with LoRA adapters...")
    trainer.train()
    
    count_params(trainer.model, "PEFT-wrapped after training")

    trainer.model = model.merge_and_unload()
    trainer.tokenizer = tokenizer

    count_params(trainer.model, "Merged final model")
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.push_to_hub(commit_message=commit_message)
    print("Training completed")


def preprocess(example,
               tokenizer,
               MAX_LENGTH,
               tokenized_marker):

    def find_last_subseq(seq, pat):
        for i in range(len(seq) - len(pat), -1, -1):
            if seq[i : i+len(pat)] == pat:
                return i + len(pat) - 1
        raise ValueError("Could not find marker")

    
    topic = "knowledge and skills in advanced master-level STEM courses"
    lines = [
        f"The following are multiple choice questions (with answers) about {topic}.",
        "",
        example["question"],
    ]
    for label, option in zip(["A","B","C","D","E"], example["options"]):
        lines.append(f"{label}. {option}")
    lines.append("Answer:")
    prompt = "\n".join(lines)

    choice_ids = tokenizer.encode(f" {example['label']}", add_special_tokens=False)
    n_answer = len(choice_ids)

    tok = tokenizer(
        prompt,
        max_length=MAX_LENGTH - n_answer,
        truncation=True,
        padding=False,
    )

    pad_len = MAX_LENGTH - len(tok["input_ids"])
    tok["input_ids"]      += [tokenizer.pad_token_id] * pad_len
    tok["attention_mask"] += [0] * pad_len

    labels = [-100] * MAX_LENGTH
    ans_pos = find_last_subseq(tok["input_ids"], tokenized_marker)
    if ans_pos + 1 + n_answer > MAX_LENGTH:
        raise ValueError("Not enough room for answer tokens")
    for i, tid in enumerate(choice_ids):
        labels[ans_pos + 1 + i] = tid

    return {
        "input_ids":      tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "labels":         labels,
    }

# ─── Main ──────────────────────────────────────────

if __name__ == "__main__":
    fine_tune_peft_model(
        preprocess,
        MY_MODEL_NAME = 'test',
        DATA_SUBSET = 50
    )