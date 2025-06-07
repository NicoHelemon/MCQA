from unsloth import FastModel
import os
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel
from huggingface_hub import HfApi

from transformers import TrainerCallback, TrainerState, TrainerControl
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, update_display
from transformers.trainer_callback import PrinterCallback
import json

class LivePlotCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.train_steps  = []
        self.steps_per_epoch = None
        self.display_id      = "train_loss_plot"  # constant string id
        self._first_display  = True               # flag
        self.printer         = PrinterCallback()

    def on_train_begin(self, args, state, control, **kwargs):
        # Determine number of batches per epoch
        train_dataloader = kwargs.get('train_dataloader')
        if train_dataloader is not None and hasattr(train_dataloader, '__len__'):
            self.steps_per_epoch = len(train_dataloader)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        # Collect training loss
        if 'loss' in logs:
            loss = logs['loss']
            self.train_losses.append(loss)
            self.train_steps.append(state.global_step)

        # Redraw plot
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot training loss
        if self.train_losses:
            ax.plot(self.train_steps, self.train_losses, label='Train Loss')
            ax.set_ylabel('Loss')
            # Annotate latest loss value
            latest_idx = self.train_steps[-1] - 1
            latest_loss = self.train_losses[-1]
            ax.annotate(f'{latest_loss:.2f}', xy=(latest_idx, latest_loss),
                        xytext=(0, 5), textcoords='offset points', ha='center')

        # # Annotate epoch progress fraction (current_epoch / total_epochs)
        # if self.steps_per_epoch:
        #     current_epoch = state.global_step / self.steps_per_epoch
        #     total_epochs = getattr(args, 'num_train_epochs', None)
        #     if total_epochs:
        #         ax.text(0.8, 0.9, f'Epoch: {current_epoch:.2f}/{total_epochs}\nProgress: {current_epoch/total_epochs:.2f}',
        #                 transform=ax.transAxes, ha='right', va='top')

        ax.set_xlabel('Steps')
        ax.set_ylim(0, 2)
        ax.set_title('Training Loss')
        #ax.legend(loc='upper right')
        ax.grid(False)

        if self._first_display:
            display(fig, display_id=self.display_id)
            self._first_display = False
        else:
            update_display(fig, display_id=self.display_id)

        # save a copy to disk
        os.makedirs(args.output_dir, exist_ok=True)
        filename = os.path.join(args.output_dir, f"training_loss.png")
        fig.savefig(filename)
        # close to prevent the “static” matplotlib output
        plt.close(fig)

        return control


class QwenMCQA:
    def __init__(
        self,
        preprocess,
        #compute_metrics,
        my_model_suffix: str = "",
        num_epochs: int = 1,
        save_steps: int = 50,
        lora_r: int = 128,
        lora_alpha: int = 256,
        data_subset: int = 0,
        commit_message: str = "new commit",
        hf_token: str = "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",
        hf_username: str = "NicoHelemon",
        base_model: str = "Qwen/Qwen3-0.6B-Base",
        max_length: int = 512,
        batch_size: int = 256,
        learning_rate: float = 3e-5,
        logging_steps: int = 10,
        gradient_accumulation_steps: int = 1,
        bf16: bool = True,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "cosine",
    ):
        # Model naming
        if my_model_suffix and not my_model_suffix.startswith("_"):
            my_model_suffix = "_" + my_model_suffix
        self.model_name = my_model_suffix

        # Environment & identifiers
        os.environ["HF_TOKEN"] = hf_token
        self.hf_token = hf_token
        self.hf_username = hf_username
        self.repo_ds = f"{hf_username}/MNLP_M2_mcqa_dataset"
        self.hub_model_id = f"{hf_username}/MNLP_M2_mcqa_model{my_model_suffix}"
        self.base_model = base_model
        self.output_dir = f"tmp/mcqa_model{my_model_suffix}"
        self.logging_dir = f"{self.output_dir}/logs"

        # Hyperparameters
        self.num_epochs = num_epochs
        self.save_steps = save_steps
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.data_subset = data_subset
        self.commit_message = commit_message

        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.logging_steps = logging_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.bf16 = bf16
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type

        # User-provided preprocessing function
        self.preprocess = preprocess

        # Prepare all components
        self._setup()  # load data, model/tokenizer, dataset, trainer

    def _load_data(self):
        raw     = load_dataset(self.repo_ds)
        if self.data_subset > 0:
            r = raw["train"].shuffle(seed=42).select(range(self.data_subset))
            v = raw["validation"].shuffle(seed=42).select(range(self.data_subset // 10))
            return r, v
        r = raw["train"].shuffle(seed=42)
        v = raw["validation"].shuffle(seed=42).select(range(500))
        return r, v

    def _setup_model_tokenizer(self):
        model, tokenizer = FastModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_length,
            token=self.hf_token,
        )
        model.gradient_checkpointing_enable()

        model = FastModel.get_peft_model(
            model,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "up_proj","down_proj",
                "layer_norm1","layer_norm2"
            ],
            bias="all",
        )

        tokenizer.truncation_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    def _prepare_dataset(self, train_raw, tokenizer):
        tokenized_marker = tokenizer("Answer:").input_ids
        return train_raw.map(
            self.preprocess,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": self.max_length,
                "tokenized_marker": tokenized_marker
            },
            remove_columns=train_raw.column_names,
        )

    def _create_trainer(self, model, tokenizer, train_ds, val_ds):        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            push_to_hub=True,
            hub_model_id=self.hub_model_id,
            hub_token=self.hf_token,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim="adamw_torch",
            bf16=self.bf16,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_strategy="steps",
            #logging_dir=self.logging_dir,
            #eval_strategy="steps",
            #eval_steps=self.save_steps,
            #report_to=[LivePlotCallback()],
            #load_best_model_at_end=True,
            #metric_for_best_model ="accuracy"
        )
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=default_data_collator,
            callbacks=[LivePlotCallback()]
            #eval_dataset=val_ds,
            #compute_metrics=lambda eval_pred : self.compute_metrics(eval_pred, tokenizer),
        )

    def _setup(self):
        # Pure setup: load data and build trainer
        self.train_raw, self.val_raw = self._load_data()
        self.model, self.tokenizer = self._setup_model_tokenizer()
        self.train_ds = self._prepare_dataset(self.train_raw, self.tokenizer)
        self.val_ds = self._prepare_dataset(self.val_raw, self.tokenizer)
        self.trainer = self._create_trainer(
            self.model, self.tokenizer, self.train_ds, self.val_ds
        )

    def _get_last_checkpoint(self, before_step = None):
        """
        Scan self.output_dir for folders named 'checkpoint-<step>' and
        return the path to the one with the highest <step>.
        If before_step is given, only consider checkpoints with step < before_step.
        """
        if not os.path.isdir(self.output_dir):
            return None

        ckpts = []
        for d in os.listdir(self.output_dir):
            full = os.path.join(self.output_dir, d)
            if os.path.isdir(full) and d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-", 1)[1])
                except ValueError:
                    continue
                if before_step is None or step < before_step:
                    ckpts.append((step, full))

        if not ckpts:
            return None

        # return the path corresponding to the max step
        return max(ckpts, key=lambda x: x[0])[1]

    def _is_terminal_checkpoint(self, ckpt: str) -> bool:
        """
        Return True if ckpt/trainer_state.json says we've already
        completed all epochs.
        """
        state_file = os.path.join(ckpt, "trainer_state.json")
        if not os.path.isfile(state_file):
            return False
        with open(state_file, "r") as f:
            st = json.load(f)
        # when epoch ≥ num_train_epochs, this checkpoint is final
        return st.get("epoch", 0) >= st.get("num_train_epochs", float("inf"))

    def train(self, resume: bool = True, from_ckpt: int = 0):
        print("Starting training with LoRA adapters...")
        if resume:
            # pass from_ckpt (0 means no threshold)
            threshold = from_ckpt if from_ckpt > 0 else None
            ckpt = self._get_last_checkpoint(before_step=threshold)
        else:
            ckpt = None

        if ckpt:
            print(f"  found checkpoint {ckpt}")
            if self._is_terminal_checkpoint(ckpt):
                print("✅  Last checkpoint is terminal. Skipping training.")
            else:
                print("⏩  Resuming training from that checkpoint …")
                self.trainer.train(resume_from_checkpoint=ckpt)
        else:
            print("🔄  No checkpoint found, starting from scratch …")
            self.trainer.train()

        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.trainer.push_to_hub(commit_message=self.commit_message)
        print("Training completed")


    def merge(self, new_model_suffix: str = None):
        """
        If new_model_suffix is None:
          → Exactly replicate your old behavior:
            • Load adapter from self.hub_model_id  
            • Merge it into base  
            • Save under self.output_dir  
            • Call self.trainer.push_to_hub() so it pushes self.output_dir → self.hub_model_id

        If new_model_suffix is not None:
          → Load adapter from the ORIGINAL self.hub_model_id  
          → Merge into base  
          → Save merged files under tmp/mcqa_model_<suffix>/  
          → Explicitly push merged weights + tokenizer into 
            "NicoHelemon/MNLP_M2_mcqa_model_<suffix>"  
        """

        # 1) Decide “source” vs. “target” based on whether new_model_suffix was provided
        if new_model_suffix is None:
            # (a) No suffix: keep exactly the old repo & local dir
            source_hub_id     = self.hub_model_id
            target_output_dir = self.output_dir
            target_hub_id     = self.hub_model_id
        else:
            # (b) We want to merge from the old adapter, but push into a new repo
            if not new_model_suffix.startswith("_"):
                new_model_suffix = "_" + new_model_suffix

            source_hub_id     = self.hub_model_id
            target_hub_id     = f"{self.hf_username}/MNLP_M2_mcqa_model{new_model_suffix}"
            target_output_dir = f"tmp/mcqa_model{new_model_suffix}"

        # helper to check parameter counts
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        print("🔄 Merging LoRA adapters into base model…")

        # 2) Load base model + tokenizer
        tokenizer  = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model, torch_dtype="auto")

        # 3) Attach the adapter from source_hub_id (always the original self.hub_model_id)
        print(f"📥  Loading LoRA adapter from '{source_hub_id}' …")
        adapter_wrapped = PeftModel.from_pretrained(
            base_model,
            source_hub_id,
            torch_dtype="auto"
        )

        # 4) Merge LoRA weights into the base model
        print("🔄  Performing merge_and_unload()…")
        merged = adapter_wrapped.merge_and_unload()
        assert count_parameters(adapter_wrapped.base_model) == count_parameters(merged)

        # 5) Remove any quantization_config so HF won’t complain
        if hasattr(merged.config, "quantization_config"):
            del merged.config.quantization_config

        # 6) Save merged model + tokenizer locally under target_output_dir
        os.makedirs(target_output_dir, exist_ok=True)
        print(f"💾  Saving merged model into '{target_output_dir}' …")
        merged.save_pretrained(target_output_dir)
        tokenizer.save_pretrained(target_output_dir)

        # 7) Push to the correct Hugging Face repo
        if new_model_suffix is None:
            # EXACTLY the old behavior: let the Trainer push from self.output_dir → self.hub_model_id
            print(f"🚀  Pushing '{target_output_dir}' → '{target_hub_id}' via self.trainer.push_to_hub …")
            self.trainer.push_to_hub(
                commit_message=f"✅ Merged model saved to '{target_output_dir}' and pushed to '{target_hub_id}'"
            )
        else:
            # We must push merged + tokenizer manually to the new repo name
            print(f"🚀  Pushing merged weights → '{target_hub_id}' …")
            merged.push_to_hub(
                repo_id=target_hub_id,
                use_auth_token=self.hf_token,
                commit_message=f"✅ Merged model (truncated) pushed to '{target_hub_id}'"
            )
            tokenizer.push_to_hub(
                repo_id=target_hub_id,
                use_auth_token=self.hf_token,
                commit_message=f"✅ Tokenizer for merged model pushed to '{target_hub_id}'"
            )

        print(f"✅  Merge complete. Local directory: '{target_output_dir}', Hub repo: '{target_hub_id}'")


    def push_truncated_model(self, before_step: int, new_model_suffix: str, commit_message: str = "Truncated checkpoint push"):
        """
        Load the most recent checkpoint whose step < before_step, then push that
        partially trained (adapter-wrapped) model to the Hub under a new repo name.

        Args:
            before_step (int): Only consider checkpoints with a global_step < before_step.
            new_model_suffix (str): Suffix to append to `MNLP_M2_mcqa_model` for the new repo.
                                    (e.g. if new_model_suffix="cp_1000", the Hub ID becomes
                                    "<hf_username>/MNLP_M2_mcqa_model_cp_1000")
            commit_message (str): Message to use when pushing to the Hub.
        """
        # 1) Find the checkpoint folder
        ckpt = self._get_last_checkpoint(before_step=before_step)
        if ckpt is None:
            raise ValueError(f"No checkpoint found before step {before_step}")

        print(f"🔍  Found checkpoint: {ckpt}")

        # 2) Instantiate base model and then load adapter weights
        #    (this assumes the checkpoint contains both config.json and adapter files)
        print("🔄  Loading base model and LoRA adapters from checkpoint…")
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model, torch_dtype="auto")
        model = PeftModel.from_pretrained(base_model, ckpt, torch_dtype="auto")

        # 3) Load tokenizer (same as base)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)

        # 4) Define new Hub repo ID
        if new_model_suffix and not new_model_suffix.startswith("_"):
            new_model_suffix = "_" + new_model_suffix
        new_hub_model_id = f"{self.hf_username}/MNLP_M2_mcqa_model{new_model_suffix}"

        # 5) Push both model and tokenizer to the Hub
        print(f"🚀  Pushing adapter-wrapped model to '{new_hub_model_id}' …")
        model.push_to_hub(
            repo_id=new_hub_model_id,
            use_auth_token=self.hf_token,
            commit_message=commit_message
        )
        tokenizer.push_to_hub(
            repo_id=new_hub_model_id,
            use_auth_token=self.hf_token,
            commit_message=commit_message
        )

        print(f"✅  Checkpoint successfully pushed as '{new_hub_model_id}'")

