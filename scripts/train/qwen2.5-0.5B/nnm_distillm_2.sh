#!/bin/bash



# --- Accelerate launch ---
CUDA_VISIBLE_DEVICES=1 accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  src/run_nnm_distillm_math.py \
  ./training_configs/qwen2.5-0.5B/nnmdistillm2.yaml
