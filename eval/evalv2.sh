#!/bin/bash
# ============================================================
# Evaluation Script: MBPP + GSM8K + MATH500
# Repo: bigcode-project/bigcode-evaluation-harness
# ============================================================
# Usage:
#   bash run_eval.sh --model <MODEL_NAME_OR_PATH> [--tasks mbpp,gsm8k,math500]
# ============================================================

set -e

# ---------- Default args ----------
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TASKS="mbpp,gsm8k,math500"
MAX_LENGTH=2048
BATCH_SIZE=4
OUTPUT_DIR="./eval_results"

# ---------- Parse args ----------
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)   MODEL="$2";  shift 2 ;;
    --tasks)   TASKS="$2";  shift 2 ;;
    --max_len) MAX_LENGTH="$2"; shift 2 ;;
    --bs)      BATCH_SIZE="$2"; shift 2 ;;
    --output)  OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "$MODEL" ]; then
  echo "ERROR: --model is required"
  echo "Usage: bash run_eval.sh --model <MODEL_NAME_OR_PATH>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "=========================================="
echo "Model : $MODEL"
echo "Tasks : $TASKS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# ============================================================
# 1. Setup: clone & install bigcode-evaluation-harness
# ============================================================
setup_repo() {
  if [ ! -d "bigcode-evaluation-harness" ]; then
    echo "[SETUP] Cloning bigcode-evaluation-harness..."
    git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
  fi
  cd bigcode-evaluation-harness
  pip install -e . -q
  cd ..
}

# ============================================================
# 2. MBPP  (pass@1, n=15, temp=0.1)
# ============================================================
run_mbpp() {
  echo ""
  echo "========== [1/3] MBPP =========="
  cd bigcode-evaluation-harness

  accelerate launch main.py \
    --model "$MODEL" \
    --max_length_generation "$MAX_LENGTH" \
    --tasks mbpp \
    --temperature 0.1 \
    --n_samples 15 \
    --batch_size "$BATCH_SIZE" \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "../${OUTPUT_DIR}/mbpp_generations.json" \
    --metric_output_path "../${OUTPUT_DIR}/mbpp_results.json"

  echo "[DONE] MBPP results saved to ${OUTPUT_DIR}/mbpp_results.json"
  cd ..
}

# ============================================================
# 3. GSM8K via PAL  (greedy, pass@1)
#    task name: pal-gsm8k-greedy
#    NOTE: max_length >= 2048 vì prompt few-shot ~1500 tokens
# ============================================================
run_gsm8k() {
  echo ""
  echo "========== [2/3] GSM8K (PAL) =========="
  cd bigcode-evaluation-harness

  accelerate launch main.py \
    --model "$MODEL" \
    --max_length_generation "$MAX_LENGTH" \
    --tasks pal-gsm8k-greedy \
    --n_samples 1 \
    --batch_size 1 \
    --do_sample False \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "../${OUTPUT_DIR}/gsm8k_generations.json" \
    --metric_output_path "../${OUTPUT_DIR}/gsm8k_results.json"

  echo "[DONE] GSM8K results saved to ${OUTPUT_DIR}/gsm8k_results.json"
  cd ..
}

# ============================================================
# 4. MATH500
#    KHÔNG có sẵn trong bigcode-evaluation-harness
#    → Dùng lighteval hoặc script riêng bên dưới
# ============================================================
run_math500() {
  echo ""
  echo "========== [3/3] MATH500 =========="

  # Option A: dùng lighteval (khuyến nghị)
  if python -c "import lighteval" 2>/dev/null; then
    echo "[MATH500] Using lighteval..."
    lighteval accelerate \
      --model_args "pretrained=${MODEL}" \
      --tasks "lighteval|math:500|0|0" \
      --output_dir "${OUTPUT_DIR}/math500_lighteval"
    echo "[DONE] MATH500 results saved to ${OUTPUT_DIR}/math500_lighteval"

  # Option B: script riêng với datasets + vllm/transformers
  else
    echo "[MATH500] lighteval not found, running custom script..."
    pip install datasets transformers -q
    python3 - <<'PYEOF'
import json, re, os, sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = os.environ.get("EVAL_MODEL", "")
OUTPUT_PATH = os.environ.get("MATH500_OUTPUT", "./eval_results/math500_results.json")

# Load MATH-500 (subset của MATH benchmark)
dataset = load_dataset("HendrycksTest/math", split="test")
# Lọc 500 samples (MATH500 = 500 problems từ test set)
dataset = dataset.select(range(min(500, len(dataset))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

def extract_answer(text):
    """Trích đáp số cuối cùng từ \\boxed{...}"""
    match = re.findall(r"\\boxed\{([^}]+)\}", text)
    return match[-1].strip() if match else text.strip().split()[-1]

def normalize(ans):
    return ans.replace(" ", "").replace(",", "").lower()

correct = 0
results = []

FEW_SHOT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}.\n\n"
)

for i, sample in enumerate(dataset):
    prompt = FEW_SHOT + "Problem: " + sample["problem"] + "\nSolution:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = extract_answer(generated)
    gold = extract_answer(sample["solution"])

    is_correct = normalize(pred) == normalize(gold)
    if is_correct:
        correct += 1

    results.append({"id": i, "pred": pred, "gold": gold, "correct": is_correct})

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/500] Accuracy so far: {correct/(i+1)*100:.2f}%")

accuracy = correct / len(dataset)
summary = {"accuracy": accuracy, "correct": correct, "total": len(dataset)}
print(f"\n[MATH500] Final Accuracy: {accuracy*100:.2f}%")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2)
print(f"[DONE] MATH500 results saved to {OUTPUT_PATH}")
PYEOF
  fi
}

# ============================================================
# Main: chạy từng task được chọn
# ============================================================
setup_repo

IFS=',' read -ra TASK_LIST <<< "$TASKS"
for TASK in "${TASK_LIST[@]}"; do
  case "$TASK" in
    mbpp)    run_mbpp ;;
    gsm8k)   run_gsm8k ;;
    math500) EVAL_MODEL="$MODEL" MATH500_OUTPUT="${OUTPUT_DIR}/math500_results.json" run_math500 ;;
    *)       echo "WARNING: Unknown task '$TASK', skipping." ;;
  esac
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "All done! Results in: $OUTPUT_DIR/"
echo "------------------------------------------"
for f in "$OUTPUT_DIR"/*.json; do
  [ -f "$f" ] || continue
  echo "  $f"
  python3 -c "
import json, sys
with open('$f') as fp:
    d = json.load(fp)
# In key metrics nếu có
for k in ['pass@1','accuracy','em','exact_match']:
    if k in d:
        print(f'    {k}: {d[k]}')
" 2>/dev/null || true
done
echo "=========================================="