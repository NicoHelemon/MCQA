
Run pip install -r requirements.txt in a fresh env

!! Change these to yours in training.ipynb !!
hf_token: = ...
hf_username = ...
repo_ds = ...

!! Please go in QwenMCQA.py file and changes these to yours !!
Environment & identifiers
os.environ["HF_TOKEN"] = hf_token
self.hf_token = hf_token
self.hf_username = hf_username
self.repo_ds = f"{hf_username}/MNLP_M2_mcqa_dataset" if repo_ds is None else repo_ds
self.hub_model_id = f"{hf_username}/MNLP_M2_mcqa_model{my_model_suffix}"
self.base_model = base_model
self.output_dir = f"tmp/mcqa_model{my_model_suffix}"
self.logging_dir = f"{self.output_dir}/logs"

Caution, some elements and descriptions below might not be up-to-date.


**Model Overview**
This system fine-tunes a Qwen 3-0.6B language model for multi-choice QA using parameter-efficient adapters (LoRA) and optional chain-of-thought (CoT) prompting. It balances training speed, memory efficiency, and answer accuracy by:

* Wrapping the base model with lightweight LoRA adapters (r=128, α=256, dropout=0.05) on attention projections and layer norms
* Using bfloat16 mixed-precision and gradient checkpointing to reduce memory footprint
* Dynamically generating prompts that sometimes include a “think-step-by-step” rationale

---

### Prompting & Data-Processing Strategy

1. **Prompt Templates**

   * Randomly samples one of three intros (dataset-aware or generic).
   * With probability *p* (here 1.0), inserts a CoT block `<think>…</think>` containing the ground-truth rationale.
2. **Answer Labeling**

   * Enumerative labels (e.g. “A.”, “B)”, “(C)”, “D –”) are generated per question, enforcing consistent styling per dataset.
   * Only the tokens corresponding to the correct answer are used in the loss (all others masked to –100).
3. **Truncation & Padding**

   * Prompts truncated to `max_length – answer_length` to guarantee space for the gold answer tokens.
   * Sequences padded to `max_length` with the model’s EOS token.

---

### LoRA Adapter Configuration

* **Rank (r):** 128
* **Alpha:** 256 (controls LoRA update scaling)
* **Target Modules:**

  * Query/key/value/output projections (`q_proj`,`k_proj`,`v_proj`,`o_proj`)
  * Feed-forward low-rank updates (`up_proj`,`down_proj`)
  * Both layer norms (`layer_norm1`,`layer_norm2`)
* **Bias:** “all” (adapts all bias terms)
* **Merge & Unload:** After training, adapters are fused back into the base weights for efficient inference.

---

### Training Setup

* **Base Model:** `Qwen/Qwen3-0.6B` via `unsloth.FastModel`
* **Precision:** bfloat16
* **Optimizer:** AdamW (Torch)
* **Scheduler:** Cosine warmup & decay (`lr_scheduler_type="cosine"`, `warmup_ratio=0.1`)
* **Batch Size:** 256 examples per device
* **Gradient Accumulation:** 1
* **Learning Rate:** 3 × 10⁻⁵
* **Weight Decay:** 0.01
* **Num Epochs:** 1 (configurable)
* **Logging & Checkpoints:**

  * Logs every 10 steps, saves every 50 steps
  * Pushes model automatically to Hugging Face Hub under `${username}/MNLP_M2_mcqa_model_cot10`

---

### Custom Live-Plot Callback

* **LivePlotCallback:**

  * Hooks into `on_log` to capture training loss vs. global step
  * Renders a real-time plot in Jupyter, updating via `display_id`
  * Persists `training_loss.png` to the output directory on each update

---

### Workflow Summary

1. **Data Loading:** Fetches `${username}/MNLP_M2_mcqa_dataset`, optionally subsets for quick runs.
2. **Model & Tokenizer:** Loads FastModel, applies LoRA, configures padding/truncation.
3. **Dataset Prep:** Maps raw examples through `preprocess()`, masking all non-answer tokens.
4. **Trainer Creation:** Instantiates `Trainer` with our args and `LivePlotCallback`.
5. **Training:** Runs one (or more) epochs, merges adapters, saves & pushes the final model.

_______________________________


Here’s a pedagogical walkthrough of the `QwenMCQA` setup and strategy, breaking down its key components and design choices:

---

## 1. Overall Strategy

1. **LoRA-based fine-tuning**
   Rather than updating all \~600 M parameters of the Qwen3-0.6B model, we inject **Low-Rank Adapters (LoRA)** into specific attention and feed-forward modules. This drastically reduces the number of trainable parameters (only a few million) and lets us fine-tune quickly on a modest GPU.

2. **Chain-of-Thought prompting**
   During preprocessing, we probabilistically insert a “Think step by step…” rationale block when an example has a gold rationale. This encourages the model to learn intermediate reasoning patterns, not just direct answer mapping.

3. **Dynamic label formatting**
   We randomly vary how multiple-choice options are labeled—uppercase “A.”, lowercase “b)”, digits “1.”, or even bullets. This augments robustness to different MCQ styles at inference time.

---

## 2. Data Loading & Subsetting

* **Dataset**: Loaded from a user-hosted Hub dataset `NicoHelemon/MNLP_M2_mcqa_dataset`, which has train/validation splits.
* **Subsetting**: If `data_subset > 0`, takes a small slice of training and validation for rapid experiments.

---

## 3. Prompt Engineering & Preprocessing

```python
prompt, golden_answer = prompt_creator(example, cot_prob)
```

* **“Setup” sentence**
  E.g. “The following are multiple choice questions (with answers) about advanced STEM topics.”
* **Optional CoT block**
  With probability `cot_prob`, inserts:

  ```
  Solve step-by-step.
  <think>
  …gold rationale…
  </think>
  ```
* **Answer marker**
  We reserve space to append “ Answer:” plus the one-token answer. Labels are generated via `generate_labels()`, ensuring the token for the correct answer is the only one with a non-`-100` loss.

---

## 4. LoRA Adapter Injection

```python
model = FastModel.get_peft_model(
    model,
    r=self.lora_r,             # rank of adapter updates
    lora_alpha=self.lora_alpha,# scaling factor
    target_modules=[           # which modules to wrap
        "q_proj","k_proj","v_proj","o_proj",
        "up_proj","down_proj",
        "layer_norm1","layer_norm2"
    ],
    bias="all",
)
```

* **Rank (`r=128`)** & **alpha (`256`)**: control the capacity and learning speed of the adapters.
* **Target modules**: Both attention projections and feed-forward “up/down” layers, plus layer-norm biases, so each transformer block can adapt flexibly.

---

## 5. Training Setup

* **Batch size**: `256` (across GPUs or via gradient accumulation if needed)
* **Learning rate**: `3e-5` with a **cosine** scheduler + linear **warmup** (10% of total steps)
* **Precision**: **bfloat16** for memory efficiency and speed
* **Optimizer**: `adamw_torch` with weight decay `0.01`
* **Hub integration**:

  * Automatic pushing of checkpoints to `hf://NicoHelemon/MNLP_M2_mcqa_model_cot10`
  * Saves LoRA adapters and merged model transparently.

---

## 6. Live Monitoring with `LivePlotCallback`

* **Real-time Jupyter plot** of training loss vs. global step.
* **Auto-saving** of `training_loss.png` after every log.
* Uses HF’s `TrainerCallback` API to hook into `on_log`.

---

## 7. Finalization

1. **Merge** LoRA adapters back into the base weights (`model.merge_and_unload()`), producing a single “fully adapted” model.
2. **Save** locally and **push** to Hub with a custom commit message.

---

## Key Specificities & Takeaways

* **Efficient Adaptation**: LoRA lets you fine-tune a 600 M-parameter model as if it were a <10 M-parameter one.
* **Pedagogical Prompts**: Randomized CoT improves reasoning generalization.
* **Robust MCQ Formatting**: Dynamic label styles guard against over-fitting to one quiz format.
* **Live Feedback**: A built-in plotting callback keeps you informed of training dynamics without manual plotting code.
* **Seamless Hub Workflow**: Everything from dataset loading to model upload is end-to-end within the Trainer.

This design offers a blueprint for fast, robust fine-tuning of LLMs on structured tasks like multiple-choice QA, blending modern adapter methods with prompt-engineering best practices.