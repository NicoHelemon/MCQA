## Training

### 1. Create a fresh environment and install dependencies

```bash
# create & activate
conda create -n mcqa python=3.10
conda activate mcqa

# install everything you need for training
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

1. find every `checkpoint-*` folder in your `output_dir`  
2. load that adapter checkpoint and merge its LoRA weights into the base model  
3. save the merged model & tokenizer under `merged_e{epoch}`  
4. push each merged model to the Hugging Face Hub as `your-hub-model-name_e{epoch}`  

That way you end up with one Hub repo per epoch, all ready to go.  
