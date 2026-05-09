#!/bin/bash

# --- GPU selection ---
export CUDA_VISIBLE_DEVICES=0,1

# --- Accelerate launch ---
python generate/generate.py \
  --model Qwen/Qwen2.5-Math-1.5B-Instruct \
  --output_dir data/dpo/Qwen/Qwen2.5-Math-1.5B-Instruct/ \
  --batch_size 32 \
  --split train



python generate/generate.py \
  --model Qwen/Qwen2.5-0.5B \
  --output_dir data/dpo/Qwen/Qwen2.5-0.5B/ \
  --batch_size 32 \
  --split train