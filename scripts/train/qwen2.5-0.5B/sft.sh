#!/bin/bash



# --- Accelerate launch ---
accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  src/run_sft.py \
  ./training_configs/qwen2.5-0.5B/sft.yaml