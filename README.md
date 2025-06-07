## Training

# 1. Create a fresh environment and install dependencies

### install everything you need for training
```
pip install -r train_requirements.txt
```

### 2. Open and configure `models/training_M3.ipynb`

Inside that notebook you’ll see something like:

```python
from functools import partial
from your_package import preprocess, cot_prompt, QwenMCQA

# 1) how often to sample CoT vs direct
p = 1
prompt_function = cot_prompt

preprocess_f = partial(
    preprocess,
    prompt_function=prompt_function,
    cot_prob=p
)

# 2) instantiate the trainer -- change to your need
qwenMCQA = QwenMCQA(
    preprocess_f,
    # my_model_name = "MNLP_M3_mcqa_model"   # <--- change
    my_model_suffix = "cot00_b32",            # <--- change 
    num_epochs = 3,
    # learning_rate = 1e-4,                  # <--- uncomment & override if desired
    batch_size = 32,
    gradient_accumulation_steps = 1,
    hf_token = "YOUR_HF_TOKEN",               # <--- put your HF token here
    hf_username = "YOUR_HF_USERNAME"          # <--- or your HF user/org name
)
```

Be sure to replace

- `my_model_suffix` (or `my_model_name`)  
- `hf_token`  
- `hf_username`  

with whatever makes sense for your project.

### 3. Run training and merge your adapters

At the bottom of the notebook, execute:

```python
qwenMCQA.train()
qwenMCQA.merge()
```

The `merge()` step will:



## Evaluation

### 1. Set up the evaluation environment

It's recommended to use a minimal, clean environment that includes only the packages required by the course see https://github.com/eric11eca/lighteval-epfl-mnlp/blob/main/COMPUTE.md
Alternatively, you can install the needed packages from the provided file:

pip install -r lighteval-epfl-mnlp/eval_requirements.txt
```

> ⚠️ Note: The `eval_requirements.txt` file should cover everything, but there is no formal guarantee that it's strictly equivalent to the course kernel.

---

### 2. Run the evaluation script

Inside the `lighteval-epfl-mnlp` folder, run:

```bash
bash eval.sh
```

The `eval.sh` script looks like this:

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1) Define model config(s) to evaluate
MODEL_CONFIGS=(
  "model_configs/MNLP_M3_mcqa_model_haoqi_cot00_e3_quantized.yaml"
)

# 2) Define evaluation tasks and output folder
TASKS1="community|mnlp_mcqa_evals_legacy|0|0,community|mnlp_mcqa_evals|0|0,community|MNLP_M3_mcqa_dataset|0|0,community|mmlu:stem|0|0,helm|med_qa|0|0,helm|commonsenseqa|0|0,original|arc:c:letters|0|0"
TASKS2="lighteval|agieval:aqua-rat|0|0,lighteval|sciq|0|0,lighteval|openbookqa|0|0,lighteval|race:high|0|0"
TASKS3="original|mmlu|0|0"
OUTPUT_DIR="outputs"
EVAL_MODE="lighteval"

# 3) Run evaluation loop
for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
  echo "=== Running with $MODEL_CONFIG ==="
  lighteval accelerate \
    --eval-mode "${EVAL_MODE}" \
    --save-details \
    --custom-tasks "community_tasks/my_evals.py" \
    --output-dir "${OUTPUT_DIR}" \
    "${MODEL_CONFIG}" \
    "${TASKS1}"

  lighteval accelerate \
    --eval-mode "${EVAL_MODE}" \
    --save-details \
    --output-dir "${OUTPUT_DIR}" \
    "${MODEL_CONFIG}" \
    "${TASKS2}"

  lighteval accelerate \
    --eval-mode "${EVAL_MODE}" \
    --save-details \
    --output-dir "${OUTPUT_DIR}" \
    "${MODEL_CONFIG}" \
    "${TASKS3}"
done
```

To evaluate your model(s), simply **uncomment** the line corresponding to the config YAML file you want to test, and **comment out or remove the others**.

Each config must be defined inside the `model_configs/` folder and should specify model loading, tokenizer, precision, etc.

---

### 3. Aggregate and export results

Once evaluation is complete, go to:

```
lighteval-epfl-mnlp/outputs/results/
```

Open and run the notebook:

```
aggregation.ipynb
```

It will gather the individual `.json` result files and generate a CSV file summarizing evaluation metrics (raw scores and weighted means) across tasks and models.


1. find every `checkpoint-*` folder in your `output_dir`  
2. load that adapter checkpoint and merge its LoRA weights into the base model  
3. save the merged model & tokenizer under `merged_e{epoch}`  
4. push each merged model to the Hugging Face Hub as `your-hub-model-name_e{epoch}`  

That way you end up with one Hub repo per epoch, all ready to go.  
