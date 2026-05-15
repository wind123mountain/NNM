#!/bin/bash

CKPT_DIR="./ckpts"

hf download VoCuc/nnm --include "qwen2.5-1.5B-it-sft/checkpoint-1149/*" \
        --local-dir "${CKPT_DIR}"

hf download VoCuc/nnm --include "qwen2.5-1.5B-it-distillm2/checkpoint-1869/*" \
    --local-dir "${CKPT_DIR}"

hf download VoCuc/nnm --include "qwen2.5-1.5B-it-distillm2-1epoch/checkpoint-2492/*" \
    --local-dir "${CKPT_DIR}"

hf download VoCuc/nnm --include "llama-3.2-3B-it-sft/checkpoint-1911/*" \
    --local-dir "${CKPT_DIR}"

hf download VoCuc/nnm --include "llama-3.2-3B-it-distillm2/checkpoint-1869/*" \
    --local-dir "${CKPT_DIR}"

hf download VoCuc/nnm --include "llama-3.2-3B-it-distillm2-1epoch/checkpoint-2492/*" \
    --local-dir "${CKPT_DIR}"

hf download VoCuc/AMiD --include "llama3.2-3B-Instruct#amid/ab_pr_0.5_0.5_4_1e-4/7476/*" \
    --local-dir "${CKPT_DIR}"

hf download VoCuc/AMiD --include "qwen2.5-0.5B#amid/ab_pr_0.5_0.5_8_1e-4/7476/*" \
    --local-dir "${CKPT_DIR}"


TP=4

LOG_DIR="outputs/eval_results/logs"
OUT_DIR="outputs/eval_results/vllm"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"


run_eval() {
    local LABEL=$1
    local MODEL_ARGS=$2
    local OUT="${OUT_DIR}/${LABEL}"
    local LOG="${LOG_DIR}/${LABEL}.log"

    mkdir -p "${OUT}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Bắt đầu: ${LABEL} ==="

    BASE_ARGS=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --apply_chat_template
        --fewshot_as_multiturn
        --include_path "${INCLUDE_PATH}"
        --log_samples
        --output_path "${OUT}"
        --max_new_tokens 2048          # Tăng cho reasoning + code
        --temperature 0.6
        --top_p 0.95
    )

    {
        echo "=========================================="
        echo "Label: ${LABEL}"
        echo "Start: $(date)"
        echo "=========================================="

        echo ">>> [1/10] GSM8K (CoT)"
        lm_eval "${BASE_ARGS[@]}" --tasks gsm8k_cot --num_fewshot 5

        echo ">>> [2/10] MATH500 (tốt nhất)"
        lm_eval "${BASE_ARGS[@]}" --tasks hendrycks_math500 --num_fewshot 4

        echo ">>> [3/10] MMLU-STEM"
        lm_eval "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        echo ">>> [4/10] SciQ"
        lm_eval "${BASE_ARGS[@]}" --tasks sciq --num_fewshot 0

        echo ">>> [5/10] MBPP"
        lm_eval "${BASE_ARGS[@]}" --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code

        echo "=========================================="
        echo "DONE: ${LABEL} | $(date)"
        echo "=========================================="
    } 2>&1 | tee "${LOG}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Xong: ${LABEL} ==="
}

# ============================================================
# Qwen — full model local
# ============================================================

run_eval \
    "qwen2.5-1.5B-it-sft-checkpoint-1149" \
    "pretrained=${CKPT_DIR}/qwen2.5-1.5B-it-sft/checkpoint-1149,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

run_eval \
    "qwen2.5-1.5B-it-distillm2-checkpoint-1869" \
    "pretrained=${CKPT_DIR}/qwen2.5-1.5B-it-distillm2/checkpoint-1869,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

run_eval \
    "qwen2.5-1.5B-it-distillm2-1epoch-checkpoint-2492" \
    "pretrained=${CKPT_DIR}/qwen2.5-1.5B-it-distillm2-1epoch/checkpoint-2492,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"


run_eval \
    "llama-3.2-3B-it-sft-checkpoint-1911" \
    "pretrained=${CKPT_DIR}/llama-3.2-3B-it-sft/checkpoint-1911,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

run_eval \
    "llama-3.2-3B-it-distillm2-checkpoint-1869" \
    "pretrained=${CKPT_DIR}/llama-3.2-3B-it-distillm2/checkpoint-1869,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

run_eval \
    "llama-3.2-3B-it-distillm2-1epoch-checkpoint-2492" \
    "pretrained=${CKPT_DIR}/llama-3.2-3B-it-distillm2-1epoch/checkpoint-2492,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

run_eval \
    "qwen2.5-0.5B#amid-ab_pr_0.5_0.5_8_1e-4-7476" \
    "pretrained=${CKPT_DIR}/qwen2.5-0.5B#amid/ab_pr_0.5_0.5_8_1e-4/7476,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

run_eval \
    "qwen2.5-14B-Instruct" \
    "pretrained=Qwen/Qwen2.5-14B-Instruct,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

run_eval \
    "qwen2.5-Math-1.5B-Instruct" \
    "pretrained=Qwen/Qwen2.5-Math-1.5B-Instruct,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

run_eval \
    "deepseek-R1-Distill-Llama-8B" \
    "pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"


echo "=== Done ==="
