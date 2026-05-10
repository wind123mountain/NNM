#!/bin/bash



# --- Accelerate launch ---
accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  src/run_sft.py \
  ./training_configs/llama3.2-3B-it/sft.yaml