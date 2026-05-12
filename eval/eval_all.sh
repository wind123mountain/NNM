#!/bin/bash
# eval_vllm.sh — đầy đủ 10 tasks, vLLM multi-GPU
# Usage: bash eval_vllm.sh

# ============================================================
# Config
# ============================================================
export CUDA_VISIBLE_DEVICES=0,1        # chọn GPU
TP=2                                   # tensor_parallel = số GPU

VENV="/home/phongdq/projects/NNM/.venv/bin"
LM_EVAL="${VENV}/lm_eval"
INCLUDE_PATH="$(pwd)/custom_tasks"     # chứa gsm_plus.yaml
LOG_DIR="logs/eval_vllm"
OUT_DIR="results/vllm"
ADAPTER_BASE="/home/phongdq/projects/NNM/adapters"
LLAMA_BASE="meta-llama/Llama-3.2-3B-Instruct"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# ============================================================
# Hàm chạy đủ 10 tasks cho 1 model
# ============================================================
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
        --device cuda
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

        echo ">>> [2/10] MATH 500"
        "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks minerva_math500 --num_fewshot 4

        # echo ">>> [4/10] MMLU-Pro-Math"
        # "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks mmlu_pro_math --num_fewshot 5

        echo ">>> [3/10] MMLU-STEM"
        "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        echo ">>> [4/10] SciQ"
        "${LM_EVAL}" "${BASE_ARGS[@]}" --tasks sciq --num_fewshot 0

        echo ">>> [5/10] MBPP"
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

# ============================================================
# Qwen — Full model (3 models)
# ============================================================
run_eval \
    "qwen2.5-1.5B-it-sft__checkpoint-1149" \
    "pretrained=VoCuc/nnm,subfolder=qwen2.5-1.5B-it-sft/checkpoint-1149,tensor_parallel_size=${TP},dtype=bfloat16,trust_remote_code=True"

run_eval \
    "qwen2.5-1.5B-it-distillm2__checkpoint-1869" \
    "pretrained=VoCuc/nnm,subfolder=qwen2.5-1.5B-it-distillm2/checkpoint-1869,tensor_parallel_size=${TP},dtype=bfloat16,trust_remote_code=True"

run_eval \
    "qwen2.5-1.5B-it-distillm2-1epoch__checkpoint-2492" \
    "pretrained=VoCuc/nnm,subfolder=qwen2.5-1.5B-it-distillm2-1epoch/checkpoint-2492,tensor_parallel_size=${TP},dtype=bfloat16,trust_remote_code=True"

# ============================================================
# LLaMA — LoRA adapter (3 models, cần download adapter trước)
# ============================================================
run_eval \
    "llama-3.2-3B-it-sft__checkpoint-1911" \
    "pretrained=${LLAMA_BASE},peft=${ADAPTER_BASE}/llama-3.2-3B-it-sft/checkpoint-1911,tensor_parallel_size=${TP},dtype=bfloat16,trust_remote_code=True"

run_eval \
    "llama-3.2-3B-it-distillm2__checkpoint-1869" \
    "pretrained=${LLAMA_BASE},peft=${ADAPTER_BASE}/llama-3.2-3B-it-distillm2/checkpoint-1869,tensor_parallel_size=${TP},dtype=bfloat16,trust_remote_code=True"

run_eval \
    "llama-3.2-3B-it-distillm2-1epoch__checkpoint-2492" \
    "pretrained=${LLAMA_BASE},peft=${ADAPTER_BASE}/llama-3.2-3B-it-distillm2-1epoch/checkpoint-2492,tensor_parallel_size=${TP},dtype=bfloat16,trust_remote_code=True"

echo ""
echo "=========================================="
echo "TẤT CẢ DONE | $(date)"
echo "Kết quả: ${OUT_DIR}"
echo "=========================================="