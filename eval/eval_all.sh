#!/bin/bash

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[math,ifeval,sentencepiece]"
pip install langdetect immutabledict   # cho IFEval
cd ..

hf download VoCuc/nnm \
    --include "llama-3.2-3B-it-sft/checkpoint-1911/*" \
    --local-dir ./adapters

hf download VoCuc/nnm \
    --include "llama-3.2-3B-it-distillm2/checkpoint-1869/*" \
    --local-dir ./adapters

hf download VoCuc/nnm \
    --include "llama-3.2-3B-it-distillm2-1epoch/checkpoint-2492/*" \
    --local-dir ./adapters

export CUDA_VISIBLE_DEVICES=0,1   # chọn GPU muốn dùng
TP=2                               # tensor_parallel = số GPU

VENV="./.venv/bin"
LM_EVAL="${VENV}/lm_eval"
LOG_DIR="logs/eval"
OUT_DIR="results/vllm"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# ── Qwen models (full model) ──────────────────────────────
QWEN_MODELS=(
    "qwen2.5-1.5B-it-sft|checkpoint-1149"
    "qwen2.5-1.5B-it-distillm2|checkpoint-1869"
    "qwen2.5-1.5B-it-distillm2-1epoch|checkpoint-2492"
)

for ENTRY in "${QWEN_MODELS[@]}"; do
    SUBFOLDER="${ENTRY%%|*}"
    CKPT="${ENTRY##*|}"
    LABEL="${SUBFOLDER}__${CKPT}"
    echo "=== Chạy: ${LABEL} ==="

    "${LM_EVAL}" \
        --model vllm \
        --model_args "pretrained=VoCuc/nnm,subfolder=${SUBFOLDER}/${CKPT},tensor_parallel_size=${TP},dtype=bfloat16,trust_remote_code=True" \
        --tasks gsm8k,minerva_math,mmlu_stem \
        --num_fewshot 5 \
        --apply_chat_template \
        --batch_size auto \
        --output_path "${OUT_DIR}/${LABEL}" \
        2>&1 | tee "${LOG_DIR}/${LABEL}.log"
done

# ── LLaMA models (LoRA adapter — download về local trước) ─
LLAMA_BASE="meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_BASE="./adapters"

LLAMA_MODELS=(
    "llama-3.2-3B-it-sft|checkpoint-1911"
    "llama-3.2-3B-it-distillm2|checkpoint-1869"
    "llama-3.2-3B-it-distillm2-1epoch|checkpoint-2492"
)

for ENTRY in "${LLAMA_MODELS[@]}"; do
    SUBFOLDER="${ENTRY%%|*}"
    CKPT="${ENTRY##*|}"
    LABEL="${SUBFOLDER}__${CKPT}"
    ADAPTER="${ADAPTER_BASE}/${SUBFOLDER}/${CKPT}"
    echo "=== Chạy: ${LABEL} ==="

    "${LM_EVAL}" \
        --model vllm \
        --model_args "pretrained=${LLAMA_BASE},peft=${ADAPTER},tensor_parallel_size=${TP},dtype=bfloat16,trust_remote_code=True" \
        --tasks gsm8k,minerva_math,mmlu_stem \
        --num_fewshot 5 \
        --apply_chat_template \
        --batch_size auto \
        --output_path "${OUT_DIR}/${LABEL}" \
        2>&1 | tee "${LOG_DIR}/${LABEL}.log"
done

echo "=== DONE ==="