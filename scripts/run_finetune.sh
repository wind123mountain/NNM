#!/bin/bash
set -e  # Dừng nếu có lỗi

# Đường dẫn thư mục train (sửa lại cho đúng với repo của bạn)
TRAIN_DIR="./train"

echo "========================================="
echo "[1/3] Running llama3.2-3B-it/nnm_distillm_2.sh"
echo "========================================="
bash "$TRAIN_DIR/llama3.2-3B-it/nnm_distillm_2.sh"

echo "========================================="
echo "[2/3] Running qwen2.5-0.5B/nnm_distillm_2.sh"
echo "========================================="
bash "$TRAIN_DIR/qwen2.5-0.5B/nnm_distillm_2.sh"

echo "========================================="
echo "[3/3] Running qwen2.5-1.5B-it/nnm_distillm_2.sh"
echo "========================================="
bash "$TRAIN_DIR/qwen2.5-1.5B-it/nnm_distillm_2.sh"

echo "========================================="
echo "All 3 nnm_distillm_2 runs completed!"
echo "========================================="