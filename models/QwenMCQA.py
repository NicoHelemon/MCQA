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

from IPython.display import clear_output
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainerState, TrainerControl

from transformers import TrainerCallback
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, update_display
from transformers.trainer_callback import PrinterCallback

class LivePlotCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
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

        # Redraw plot
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot training loss
        if self.train_losses:
            ax.plot(self.train_losses, label='Train Loss')
            ax.set_ylabel('Loss')
            # Annotate latest loss value
            latest_idx = len(self.train_losses) - 1
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
        my_model_name: str = "",
        num_epochs: int = 1,
        save_steps: int = 50,
        lora_r: int = 128,
        lora_alpha: int = 256,
        data_subset: int = 0,
        commit_message: str = "new commit",
        hf_token: str = "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",
        hf_username: str = "NicoHelemon",
        base_model: str = "Qwen/Qwen3-0.6B",
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
        if my_model_name and not my_model_name.startswith("_"):
            my_model_name = "_" + my_model_name
        self.model_name = my_model_name

        # Environment & identifiers
        os.environ["HF_TOKEN"] = hf_token
        self.hf_token = hf_token
        self.hf_username = hf_username
        self.repo_ds = f"{hf_username}/MNLP_M2_mcqa_dataset"
        self.hub_model_id = f"{hf_username}/MNLP_M2_mcqa_model{my_model_name}"
        self.base_model = base_model
        self.output_dir = f"tmp/mcqa_model{my_model_name}"
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

    def train(self):
        print("Starting training with LoRA adapters...")
        self.trainer.train()

        # merge adapters and finalize
        merged_model = self.model.merge_and_unload()
        self.trainer.model = merged_model
        self.trainer.tokenizer = self.tokenizer

        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.trainer.push_to_hub(commit_message=self.commit_message)
        print("Training completed")
