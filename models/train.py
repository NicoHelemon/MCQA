#!/usr/bin/env python3
"""
Executable script to train and merge QwenMCQA models with configurable parameters.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from functools import partial
from QwenMCQA import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train QwenMCQA model with configurable parameters"
    )

    # Prompt configuration
    parser.add_argument(
        "--prompt_type",
        choices=["legacy", "cot"],
        default="cot",
        help="Prompt function to use: legacy or cot"
    )
    parser.add_argument(
        "--cot_prob",
        type=float,
        default=1.0,
        help="Probability of using chain-of-thought prompting when prompt_type is cot"
    )

    # Model naming and HF settings
    parser.add_argument(
        "--my_model_suffix",
        type=str,
        default="",
        help="Suffix to append to the Hugging Face model repo name"
    )
    parser.add_argument(
        "--my_model_name",
        type=str,
        default="",
        help="Explicit Hugging Face model repo name; independent from suffix"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Hugging Face access token"
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        required=True,
        help="Hugging Face username"
    )

    # Training hyperparameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--data_subset",
        type=int,
        default=0,
        help="Number of training examples to use; 0 means the full dataset"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Select prompt function
    prompt_function = legacy_prompt if args.prompt_type == "legacy" else cot_prompt

    # Create preprocessing partial
    preprocess_f = partial(
        preprocess,
        prompt_function=prompt_function,
        cot_prob=args.cot_prob
    )

    # Initialize QwenMCQA trainer with both name and suffix
    qwenMCQA = QwenMCQA(
        preprocess=preprocess_f,
        my_model_suffix=args.my_model_suffix,
        my_model_name=args.my_model_name,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        hf_token=args.hf_token,
        hf_username=args.hf_username,
        data_subset=args.data_subset
    )

    # Execute training and merging
    qwenMCQA.train()
    qwenMCQA.merge()

if __name__ == "__main__":
    main()
