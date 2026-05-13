#!/usr/bin/env python3

import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA/DoRA adapter into base model"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model path or HF repo"
    )

    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to LoRA/DoRA adapter"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged model"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model: {args.base_model}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
    )

    print("Merging adapter into base model...")

    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output}")

    merged_model.save_pretrained(args.output)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    tokenizer.save_pretrained(args.output)

    print("Done!")


if __name__ == "__main__":
    main()