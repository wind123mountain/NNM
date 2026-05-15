#!/bin/bash
# Đánh giá pkcii/distillm2-sft trên tất cả benchmark theo paper, dùng lm-evaluation-harness.
#
# Yêu cầu cài 1 lần:
#   git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
#   cd lm-evaluation-harness
#   pip install -e ".[math,ifeval,sentencepiece]"
#   pip install langdetect immutabledict   # cho IFEval
#   cd ..
#
# Sau đó copy custom_tasks/ vào nơi nào lm_eval thấy được (--include_path).

set -e

# ============================================================
# Config
# ============================================================
MODEL_NAME="/home/hungpv/projects/NND/outputs/qwen2.5-math-nnm-chosenv2/checkpoint-2492"
# SUBFOLDER="checkpoint-1140"   # nếu model huggingface có subfolder chứa pytorch_model.bin, set tên subfolder này; nếu không có subfolder, set SUBFOLDER="" (empty string)
#MODEL_NAME="Qwen/Qwen2.5-Math-1.5B-Instruct"
TOKENIZER="/home/hungpv/projects/NND/outputs/qwen2.5-math-nnm-chosenv2/checkpoint-2492"   # nếu tokenizer cùng tên với model, set giống MODEL_NAME; nếu tokenizer khác tên hoặc có subfolder khác, set tên/tokenizer path ở đây
DEVICE="cuda: 0"
DTYPE="bfloat16"
BATCH="8"           # 'auto' để lm-eval tự chỉnh; hoặc set 32, 16, ...
OUT_DIR="./eval_results/distillm2-nnm-only-distillm2"
INCLUDE_PATH="$(pwd)/custom_tasks"   # nơi chứa gsm_plus.yaml

# Common args
COMMON_ARGS=(
    --model vllm
    # --model_args "pretrained=${MODEL_NAME},subfolder=${SUBFOLDER},tokenizer=${TOKENIZER},dtype=${DTYPE},trust_remote_code=True"
    --model_args "pretrained=${MODEL_NAME},tokenizer=${TOKENIZER},dtype=${DTYPE},trust_remote_code=True"

    --device "${DEVICE}"
    --batch_size "${BATCH}"
    --apply_chat_template
    --fewshot_as_multiturn
    --include_path "${INCLUDE_PATH}"
    --log_samples
    --output_path "${OUT_DIR}"
)

mkdir -p "${OUT_DIR}"

# ============================================================
# 1. GSM8K (5-shot, multi-turn chat)
# ============================================================
echo ">>> GSM8K"
python -m lm_eval "${COMMON_ARGS[@]}" \
    --tasks gsm8k \
    --num_fewshot 5

# ============================================================
# 2. GSM-Plus (5-shot, custom task)
# ============================================================
# echo ">>> GSM-Plus"
# lm_eval "${COMMON_ARGS[@]}" \
#     --tasks gsm_plus \
#     --num_fewshot 5

# ============================================================
# 3. MATH / MATH-500 (4-shot, Minerva format)
# ============================================================
echo ">>> MATH (Minerva 4-shot)"
python -m lm_eval "${COMMON_ARGS[@]}" \
    --tasks minerva_math500 \
    --num_fewshot 4

# ============================================================
# 4. MMLU-Pro-Math (5-shot CoT, 10-way MC)
# ============================================================
echo ">>> MMLU-Pro-Math"
python -m lm_eval "${COMMON_ARGS[@]}" \
    --tasks mmlu_pro_math \
    --num_fewshot 5

# ============================================================
# 5. MMLU-STEM (5-shot, multichoice loglikelihood group)
# ============================================================
echo ">>> MMLU-STEM"
python -m lm_eval "${COMMON_ARGS[@]}" \
    --tasks mmlu_stem \
    --num_fewshot 5

# ============================================================
# 6. SciQ (0-shot)
# ============================================================
echo ">>> SciQ"
python -m lm_eval "${COMMON_ARGS[@]}" \
    --tasks sciq \
    --num_fewshot 0

# ============================================================
# 7. MBPP (0-shot, code execution)
# ============================================================
echo ">>> MBPP"
python -m lm_eval "${COMMON_ARGS[@]}" \
    --tasks mbpp \
    --num_fewshot 3 \
    --confirm_run_unsafe_code
# NB: lm-eval mbpp default là 3-shot. Nếu paper bạn đòi 0-shot, đổi --num_fewshot 0.

# ============================================================
# 8. BBH (3-shot CoT, 27 subtasks)
# ============================================================
# echo ">>> BBH (cot 3-shot)"
# lm_eval "${COMMON_ARGS[@]}" \
#     --tasks bbh_cot_fewshot \
#     --num_fewshot 3

# ============================================================
# 9. MuSR (0-shot, leaderboard variant)
# ============================================================
# echo ">>> MuSR"
# lm_eval "${COMMON_ARGS[@]}" \
#     --tasks leaderboard_musr \
#     --num_fewshot 0

# ============================================================
# 10. IFEval (0-shot, google official rule eval)
# ============================================================
# echo ">>> IFEval"
# lm_eval "${COMMON_ARGS[@]}" \
#     --tasks leaderboard_ifeval \
#     --num_fewshot 0

echo ""
echo "=========================================="
echo "Done. Kết quả ở: ${OUT_DIR}"
echo "=========================================="