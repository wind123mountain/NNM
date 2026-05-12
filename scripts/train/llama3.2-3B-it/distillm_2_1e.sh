#!/bin/bash



# --- Accelerate launch ---
accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  src/run_distillm_math.py \
  ./training_configs/llama3.2-3B-it/distillm2-1epoch.yaml