#!/bin/bash

# --- GPU selection ---
export CUDA_VISIBLE_DEVICES=0

# --- Accelerate launch ---
python generate/generate.py \
  --model Qwen/Qwen2.5-0.5B \
  --output_dir data/dpo/Qwen/Qwen2.5-0.5B/ \
  --batch_size 32 \
  --split train