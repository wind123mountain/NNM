#!/bin/bash

cd llm-evaluation-harness
pip install -e ".[math,ifeval,sentencepiece]"
pip install langdetect immutabledict   # cho IFEval
cd ..

CKPT_DIR="./ckpts"
INCLUDE_PATH="custom_tasks/"

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

VENV="./.venv/bin"
LM_EVAL="lm_eval"
LOG_DIR="logs/eval"
OUT_DIR="results/vllm"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"


run_eval() {
    local LABEL=$1
    local MODEL_ARGS=$2
    local OUT="${OUT_DIR}/${LABEL}"
    local LOG="${LOG_DIR}/${LABEL}.log"

    mkdir -p "${OUT}"

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Bắt đầu: ${LABEL} ==="

    BASE_ARGS=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size 1
        --apply_chat_template
        --fewshot_as_multiturn
        --include_path "${INCLUDE_PATH}"
        --log_samples
        --output_path "${OUT}"
    )

    {
        echo "=========================================="
        echo "Label: ${LABEL}"
        echo "Args : ${MODEL_ARGS}"
        echo "Start: $(date)"
        echo "=========================================="

        # echo ">>> [1/10] GSM8K"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks gsm8k --num_fewshot 5

        # echo ">>> [2/10] GSM-Plus"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks gsm_plus --num_fewshot 5

        # echo ">>> [3/10] MATH Minerva"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks minerva_math500 --num_fewshot 4

        # echo ">>> [4/10] MMLU-Pro-Math"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks mmlu_pro_math --num_fewshot 5

        echo ">>> [5/10] MMLU-STEM"
        "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        echo ">>> [6/10] SciQ"
        "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks sciq --num_fewshot 0

        # echo ">>> [7/10] MBPP"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code

        # echo ">>> [8/10] BBH"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks bbh_cot_fewshot --num_fewshot 3

        # echo ">>> [9/10] MuSR"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks leaderboard_musr --num_fewshot 0

        # echo ">>> [10/10] IFEval"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks leaderboard_ifeval --num_fewshot 0

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

# run_eval \
#     "llama3.2-3B-Instruct#amid-ab_pr_0.5_0.5_4_1e-4-7476" \
#     "pretrained=meta-llama/Llama-3.2-3B-Instruct,lora_local_path=${CKPT_DIR}/llama3.2-3B-Instruct#amid/ab_pr_0.5_0.5_4_1e-4/7476,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

# run_eval \
#     "qwen2.5-0.5B#amid-ab_pr_0.5_0.5_8_1e-4-7476" \
#     "pretrained=Qwen/Qwen2.5-1.5B,lora_local_path=${CKPT_DIR}/qwen2.5-1.5B-Instruct#amid/ab_pr_0.5_0.5_4_1e-4/7476,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

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
