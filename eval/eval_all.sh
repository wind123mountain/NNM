#!/bin/bash
# Chạy eval tất cả models trong VoCuc/nnm — dùng checkpoint cuối mỗi subfolder
# Usage: bash run_eval_all.sh

# ============================================================
# Config chung
# ============================================================
export CUDA_VISIBLE_DEVICES=1

MODEL_NAME="VoCuc/nnm"
DEVICE="cuda"
DTYPE="bfloat16"
BATCH="8"
BASE_OUT_DIR="results/VoCuc-nnm"
LOG_DIR="logs/eval"
INCLUDE_PATH="$(pwd)/custom_tasks"

mkdir -p "${BASE_OUT_DIR}" "${LOG_DIR}"

# ============================================================
# Danh sách checkpoint cuối mỗi subfolder
# Qwen → full model (pytorch_model.bin)
# LLaMA → LoRA adapter (adapter_model.bin) → cần base model
# ============================================================
declare -A CHECKPOINTS=(
    ["qwen2.5-1.5B-it-distillm2"]="checkpoint-1869"
    ["qwen2.5-1.5B-it-distillm2-1epoch"]="checkpoint-2492"
    ["qwen2.5-1.5B-it-sft"]="checkpoint-1149"
    ["llama-3.2-3B-it-distillm2"]="checkpoint-1869"
    ["llama-3.2-3B-it-distillm2-1epoch"]="checkpoint-2492"   
    ["llama-3.2-3B-it-sft"]="checkpoint-1911"                
)

# LLaMA dùng LoRA adapter → cần base model
LLAMA_BASE="meta-llama/Llama-3.2-3B-Instruct"

build_args() {
    local SUBFOLDER=$1
    local CHECKPOINT=$2
    local OUT_DIR=$3

    if [[ "${SUBFOLDER}" == qwen* ]]; then
        # Full model — subfolder trỏ thẳng vào checkpoint
        local FULL_SUBFOLDER="${SUBFOLDER}/${CHECKPOINT}"
        COMMON_ARGS=(
            --model hf
            --model_args "pretrained=${MODEL_NAME},subfolder=${FULL_SUBFOLDER},tokenizer=${MODEL_NAME},tokenizer_subfolder=${SUBFOLDER},dtype=${DTYPE},trust_remote_code=True"
            --device "${DEVICE}"
            --batch_size "${BATCH}"
            --apply_chat_template
            --fewshot_as_multiturn
            --include_path "${INCLUDE_PATH}"
            --log_samples
            --output_path "${OUT_DIR}"
        )
    else
        # LLaMA — LoRA adapter
        local FULL_SUBFOLDER="${SUBFOLDER}/${CHECKPOINT}"
        COMMON_ARGS=(
            --model hf
            --model_args "pretrained=${LLAMA_BASE},peft=${MODEL_NAME},peft_subfolder=${FULL_SUBFOLDER},dtype=${DTYPE},trust_remote_code=True"
            --device "${DEVICE}"
            --batch_size "${BATCH}"
            --apply_chat_template
            --fewshot_as_multiturn
            --include_path "${INCLUDE_PATH}"
            --log_samples
            --output_path "${OUT_DIR}"
        )
    fi
}

# ============================================================
# Hàm chạy tất cả tasks cho 1 model
# ============================================================
run_tasks() {
    local SUBFOLDER=$1
    local CHECKPOINT="${CHECKPOINTS[$SUBFOLDER]}"
    local LABEL="${SUBFOLDER}__${CHECKPOINT}"
    local OUT_DIR="${BASE_OUT_DIR}/${LABEL}"
    local LOG_FILE="${LOG_DIR}/${LABEL}.log"

    mkdir -p "${OUT_DIR}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Bắt đầu: ${LABEL} ==="
    echo "  Log → ${LOG_FILE}"

    build_args "${SUBFOLDER}" "${CHECKPOINT}" "${OUT_DIR}"

    {
        echo "=========================================="
        echo "Model     : ${MODEL_NAME}"
        echo "Subfolder : ${SUBFOLDER}"
        echo "Checkpoint: ${CHECKPOINT}"
        echo "Started   : $(date)"
        echo "=========================================="

        echo ">>> GSM8K"
        lm_eval "${COMMON_ARGS[@]}" --tasks gsm8k --num_fewshot 5

        echo ">>> MATH500 (4-shot)"
        lm_eval "${COMMON_ARGS[@]}" --tasks minerva_math500 --num_fewshot 4

        echo ">>> MMLU-STEM"
        lm_eval "${COMMON_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        echo ">>> SciQ"
        lm_eval "${COMMON_ARGS[@]}" --tasks sciq --num_fewshot 0

        echo ">>> MBPP"
        lm_eval "${COMMON_ARGS[@]}" --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code

        echo "=========================================="
        echo "DONE: ${LABEL} | $(date)"
        echo "=========================================="

    } 2>&1 | tee -a "${LOG_FILE}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Xong: ${LABEL} ==="
}

run_all() {
    for SUBFOLDER in \
        "qwen2.5-1.5B-it-sft" \
        "qwen2.5-1.5B-it-distillm2-1epoch" \
        "qwen2.5-1.5B-it-distillm2" \
        "llama-3.2-3B-it-sft" \
        "llama-3.2-3B-it-distillm2-1epoch" \
        "llama-3.2-3B-it-distillm2"
    do
        run_tasks "${SUBFOLDER}"
    done
}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Checkpoints sẽ chạy:"
for SF in "${!CHECKPOINTS[@]}"; do
    echo "  ${SF} → ${CHECKPOINTS[$SF]}"
done
echo ""

nohup bash -c "
$(declare -f build_args)
$(declare -f run_tasks)
$(declare -f run_all)
$(declare -p CHECKPOINTS)
MODEL_NAME='${MODEL_NAME}'
LLAMA_BASE='${LLAMA_BASE}'
DEVICE='${DEVICE}'
DTYPE='${DTYPE}'
BATCH='${BATCH}'
BASE_OUT_DIR='${BASE_OUT_DIR}'
LOG_DIR='${LOG_DIR}'
INCLUDE_PATH='${INCLUDE_PATH}'
run_all
" >> "${LOG_DIR}/master.log" 2>&1 &

MASTER_PID=$!
echo "Master PID : ${MASTER_PID}"
echo "Master log : ${LOG_DIR}/master.log"
echo ""
echo "Theo dõi:"
echo "  tail -f ${LOG_DIR}/master.log"
echo "  tail -f '${LOG_DIR}/qwen2.5-1.5B-it-distillm2__checkpoint-1869.log'"
echo ""
echo "Dừng tất cả:"
echo "  kill ${MASTER_PID}"