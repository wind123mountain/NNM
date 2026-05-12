#!/bin/bash



# --- Accelerate launch ---
accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  src/run_distillm_math.py \
  ./training_configs/qwen2.5-0.5B/distillm2.yaml