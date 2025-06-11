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
import os
import glob
import json
import random
import numpy as np
from functools import partial

def legacy_prompt(example, cot_prob = None):
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
	return prompt, example['label']

def cot_prompt(example, cot_prob = 0.5):
	subjects = {'aqua_rat' : 'math',
				'medmcqa' : 'medicine',
				'mmlu_aux' : 'STEM',
				'openbookqa' : 'STEM',
				'sciq' : 'STEM',
				'race' : 'STEM'}
	
	topic = f"knowledge and skills in advanced master-level {subjects[example['dataset']]} courses"
	setup = f'The following are multiple choice questions (with answers) about {topic}.\n'

	cot = ''
	doing_cot = random.random() < cot_prob
	if example['rationale'] != "" and doing_cot:
		cot = f'Think step by step (before answering).\n<think>\n{example["rationale"]}\n</think>'

	golden_idx = example["label_idx"]
	labels = ['A', 'B', 'C', 'D', 'E']
	golden_answer = labels[golden_idx]

	options = [f"{label}. {option}".strip() for label, option in zip(labels, example["options"])]

	cot_and_answer = [cot, 'Answer:'] if doing_cot else ['Answer:']
	lines = [setup, example["question"]] + options + cot_and_answer

	return '\n'.join(lines), golden_answer

def preprocess(example,
			   tokenizer,
			   max_length,
			   tokenized_marker,
			   prompt_function,
			   cot_prob):

	def find_last_subseq(seq, pat):
		for i in range(len(seq) - len(pat), -1, -1):
			if seq[i : i+len(pat)] == pat:
				return i + len(pat) - 1
		raise ValueError("Could not find marker")

	prompt, golden_answer = prompt_function(example, cot_prob)

	choice_ids = tokenizer.encode(f" {golden_answer}", add_special_tokens=False)
	n_answer = len(choice_ids)

	tok = tokenizer(
		prompt,
		max_length=max_length - n_answer,
		truncation=True,
		padding=False,
	)

	pad_len = max_length - len(tok["input_ids"])
	tok["input_ids"]      += [tokenizer.pad_token_id] * pad_len
	tok["attention_mask"] += [0] * pad_len

	labels = [-100] * max_length
	ans_pos = find_last_subseq(tok["input_ids"], tokenized_marker)
	if ans_pos + 1 + n_answer > max_length:
		raise ValueError("Not enough room for answer tokens")
	for i, tid in enumerate(choice_ids):
		labels[ans_pos + 1 + i] = tid

	return {
		"input_ids":      tok["input_ids"],
		"attention_mask": tok["attention_mask"],
		"labels":         labels,
	}


class QwenMCQA:
	def __init__(
		self,
		preprocess,
		#compute_metrics,
		my_model_name: str = "MNLP_M3_mcqa_model",
		my_model_suffix: str = "",
		num_epochs: int = 1,
		save_steps: int = 50,
		lora_r: int = 128,
		lora_alpha: int = 256,
		data_subset: int = 0,
		commit_message: str = "new commit",
		hf_token: str | None = None,
		hf_username: str | None = None,
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
		self.hf_token = hf_token
		self.hf_username = hf_username
		self.pushing = bool(hf_token and hf_username)
		if self.pushing:
			print(f"The model will be pushed to Hugging Face Hub as '{hf_username}/{my_model_name}{my_model_suffix}'...")
			os.environ["HF_TOKEN"] = hf_token
		else:
			print("The model will be saved locally only and not be pushed to Hugging Face Hub (no token or username provided).")
		self.repo_ds = f"NicoHelemon/MNLP_M3_mcqa_dataset"
		self.hub_model_id = f"{hf_username}/{my_model_name}{my_model_suffix}" if self.pushing else None
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
			max_seq_length=self.max_length
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
			save_strategy="epoch",
			push_to_hub=self.pushing,
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
		self.trainer.save_model(self.output_dir)
		self.tokenizer.save_pretrained(self.output_dir)
		if self.pushing:
			self.trainer.push_to_hub(commit_message=self.commit_message)
			print("Model pushed to the Hub")
		print("Training completed")


	def merge(self):
		"""
		For every checkpoint-* in output_dir:
		  • read its trainer_state.json → epoch number
		  • merge that adapter into base
		  • push it to HF as …_e{epoch}
		"""

		def count_parameters(model):
			return sum(p.numel() for p in model.parameters())

		# 1) discover all checkpoint dirs
		pattern = os.path.join(self.output_dir, "checkpoint-*")
		ckpt_dirs = glob.glob(pattern)

		# 2) build a list of (epoch, path)
		epoch_ckpts = []
		for d in ckpt_dirs:
			state_path = os.path.join(d, "trainer_state.json")
			if not os.path.isfile(state_path):
				continue
			with open(state_path, "r") as f:
				st = json.load(f)
			# round down the float to int
			epoch = int(st.get("epoch", 0))
			if epoch > 0:
				epoch_ckpts.append((epoch, d))

		# 3) sort by epoch
		epoch_ckpts.sort(key=lambda x: x[0])

		# 4) loop and merge/push each one
		for epoch, ckpt_dir in epoch_ckpts:
			print(f"\n🔄 Merging checkpoint for epoch {epoch} → {ckpt_dir}")

			# load base model & tokenizer
			tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
			base_model = AutoModelForCausalLM.from_pretrained(
				self.base_model, torch_dtype="auto"
			)

			# load adapter from checkpoint
			adapter_wrapped = PeftModel.from_pretrained(
				base_model, ckpt_dir, torch_dtype="auto"
			)

			# merge LoRA into base
			merged = adapter_wrapped.merge_and_unload()
			assert count_parameters(adapter_wrapped.base_model) == count_parameters(merged)
			if hasattr(merged.config, "quantization_config"):
				del merged.config.quantization_config

			# save locally
			merged_dir = os.path.join(self.output_dir, f"merged_e{epoch}")
			os.makedirs(merged_dir, exist_ok=True)
			merged.save_pretrained(merged_dir)
			tokenizer.save_pretrained(merged_dir)

			if self.pushing:
				# push to its own hub repo
				suffix = f"_e{epoch}"
				repo_id = f"{self.hub_model_id}{suffix}"
				print(f"🚀 Pushing merged epoch {epoch} → {repo_id}")
				merged.push_to_hub(
					repo_id=repo_id,
					use_auth_token=self.hf_token,
					commit_message=f"✅ merged epoch {epoch}"
				)
				tokenizer.push_to_hub(
					repo_id=repo_id,
					use_auth_token=self.hf_token,
					commit_message=f"✅ tokenizer for epoch {epoch}"
				)

		str_push_save = "saved, and pushed." if self.pushing else " and saved."
		print("\n✅ All epoch checkpoints merged" + str_push_save)