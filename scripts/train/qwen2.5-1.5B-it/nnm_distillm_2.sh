#!/bin/bash



# --- Accelerate launch ---
accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  src/run_nnm_distillm_math.py \
  ./training_configs/qwen2.5-1.5B-it/nnmdistillm2.yaml