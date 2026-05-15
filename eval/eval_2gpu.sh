#!/bin/bash

CKPT_DIR="./ckpts"
INCLUDE_PATH="custom_tasks/"

TP=2

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
        --batch_size auto
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

        echo ">>> [1/10] GSM8K"
        "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks gsm8k --num_fewshot 5

        # echo ">>> [2/10] GSM-Plus"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks gsm_plus --num_fewshot 5

        echo ">>> [3/10] MATH Minerva"
        "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks minerva_math500 --num_fewshot 4

        # echo ">>> [4/10] MMLU-Pro-Math"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks mmlu_pro_math --num_fewshot 5

        echo ">>> [5/10] MMLU-STEM"
        "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        echo ">>> [6/10] SciQ"
        "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks sciq --num_fewshot 0

        echo ">>> [7/10] MBPP"
        "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code

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


# python eval/merge_model.py \
#   --base_model meta-llama/Llama-3.2-3B-Instruct \
#   --adapter outputs/llama-3.2-3B-it-sft/checkpoint-1911 \
#   --output outputs/llama-3.2-3B-it-sft/checkpoint-1911

python eval/merge_model.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --adapter outputs/v2/llama-3.2-3B-it-nnm-concatenated/checkpoint-2492 \
  --output outputs/v2/llama-3.2-3B-it-nnm-concatenated/checkpoint-2492

# run_eval \
#     "llama-3.2-3B-it-sft-checkpoint-1911" \
#     "pretrained=outputs/llama-3.2-3B-it-sft/checkpoint-1911,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

# run_eval \
#     "llama-3.2-3B-it-nnm-1epoch-checkpoint-2492" \
#     "pretrained=outputs/llama-3.2-3B-it-nnm-1epoch/checkpoint-2492,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"

python eval/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/qwen2.5-1.5B-it-nnm-1epoch/checkpoint-2492 \
  --output outputs/qwen2.5-1.5B-it-nnm-1epoch/checkpoint-2492

run_eval \
    "qwen2.5-1.5B-it-sft-checkpoint-2492" \
    "pretrained=outputs/qwen2.5-1.5B-it-nnm-1epoch/checkpoint-2492,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.85,trust_remote_code=True"



echo "=== Done ==="
