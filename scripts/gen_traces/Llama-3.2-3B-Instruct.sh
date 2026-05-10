#!/bin/bash
TP_SIZE=${1:-2}

MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR="data/dpo/meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_FILE="generated_train.jsonl"

echo "Start gen traces..."
echo "Model: $MODEL_PATH"

python ./generate/generate_vllm.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --output_file $OUTPUT_FILE \
    --num_gpus $TP_SIZE

echo "Done!"