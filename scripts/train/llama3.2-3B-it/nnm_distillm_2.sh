#!/bin/bash
set -e  # Dừng script nếu có lệnh nào lỗi

CONFIG_FILE="accelerate_configs/multi_gpu.yaml"
SCRIPT="src/run_nnm_distillm_math.py"
CONFIG_DIR="./training_configs/llama3.2-3B-it"

# Run 1: chosen
# accelerate launch \
#   --config_file "$CONFIG_FILE" \
#   "$SCRIPT" \
#   "$CONFIG_DIR/nnmdistillm2_chosen.yaml"

# Run 2: concatenated_001
CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
  --config_file "$CONFIG_FILE" \
  "$SCRIPT" \
  "$CONFIG_DIR/nnmdistillm2_concatenated_001.yaml"

# Run 3: concatenated
accelerate launch \
  --config_file "$CONFIG_FILE" \
  "$SCRIPT" \
  "$CONFIG_DIR/nnmdistillm2_concatenated.yaml"

echo "All training runs completed successfully!"