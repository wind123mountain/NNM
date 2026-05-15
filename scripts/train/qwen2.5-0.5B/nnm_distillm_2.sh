#!/bin/bash

CONFIG_FILE="accelerate_configs/multi_gpu.yaml"
SCRIPT="src/run_nnm_distillm_math.py"
CONFIG_DIR="./training_configs/qwen2.5-0.5B"

# Run 1: chosen
# accelerate launch \
#   --config_file "$CONFIG_FILE" \
#   "$SCRIPT" \
#   "$CONFIG_DIR/nnmdistillm2_chosen.yaml"

# Run 2: concatenated_001
accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  src/run_nnm_distillm_math.py \
  ./training_configs/qwen2.5-0.5B/nnmdistillm2_concatenated_001.yaml

# Run 3: concatenated
accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  src/run_nnm_distillm_math.py \
  ./training_configs/qwen2.5-0.5B/nnmdistillm2_concatenated.yaml

echo "All training runs completed successfully!"