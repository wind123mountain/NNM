#!/bin/bash

# --- GPU selection ---
export CUDA_VISIBLE_DEVICES=0,1

# --- Accelerate launch ---
accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  --num_processes=2 \
  src/run_sft.py \
  ./training_configs/llama3.2-3B-it/sft.yaml