"""
Full benchmark evaluation cho Qwen2.5-Math-1.5B-Instruct (hoặc bất kỳ Qwen-style chat model).

Benchmarks (theo paper):
    - GSM8K          (5-shot, exact match)
    - GSM-Plus       (5-shot, exact match)
    - MATH-500       (4-shot, boxed-answer match)
    - MMLU-Pro-Math  (5-shot, multiple choice 10 options)
    - MMLU-STEM      (5-shot, multiple choice 4 options)
    - SciQ           (0-shot, multiple choice 4 options)
    - MBPP           (0-shot, code execution pass@1)
    - BBH            (3-shot CoT, exact match)
    - MuSR           (0-shot CoT, multiple choice)
    - IFEval         (0-shot, rule-based instruction following - subset)

Run:
    python eval_all_benchmarks.py --benchmarks gsm8k,math,mmlu_pro_math
    python eval_all_benchmarks.py --benchmarks all
"""

import os
import re
import gc
import json
import math
import string
import signal
import argparse
import subprocess
import tempfile
from typing import List, Optional, Dict, Any

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# ============================================================================
# 1. Utility: answer extraction
# ============================================================================

def extract_gsm8k_gold(text: str) -> str:
    """Lấy số sau '####' trong câu trả lời gold của GSM8K / GSM-Plus."""
    return text.split('####')[-1].strip().replace(",", "").replace("$", "")


def extract_last_number(text: str) -> Optional[str]:
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", "").replace("$", ""))
    return nums[-1] if nums else None


def extract_boxed(text: str) -> Optional[str]:
    """
    Trả về nội dung của \\boxed{...} cuối cùng, xử lý dấu ngoặc lồng nhau.
    Nếu không có thì rơi xuống số cuối cùng trong text.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return extract_last_number(text)
    i = idx + len("\\boxed{")
    depth = 1
    buf = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
            buf.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            buf.append(c)
        else:
            buf.append(c)
        i += 1
    return "".join(buf).strip()


def normalize_math_answer(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    # bỏ kí tự dạng latex không quan trọng
    for tok in ["\\,", "\\!", "\\;", "\\:", "\\ ", "$", " "]:
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.rstrip(".")
    # 0.500 -> 0.5
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except Exception:
        pass
    return s


def math_answers_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    a, b = normalize_math_answer(a), normalize_math_answer(b)
    if a == b:
        return True
    # thử so sánh số
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


def extract_choice_letter(text: str, n_choices: int = 4) -> Optional[str]:
    letters = string.ascii_uppercase[:n_choices]
    # ưu tiên \boxed{X}
    m = re.search(r"\\boxed\{\s*\(?\s*([A-Z])\s*\)?\s*\}", text)
    if m and m.group(1) in letters:
        return m.group(1)
    # answer is X / answer: X / answer is (X)
    m = re.search(r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\(?\s*([A-Z])\s*\)?",
                  text, re.IGNORECASE)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    # the answer is X.
    m = re.search(r"\bis\s+\(?([A-Z])\)?\b", text)
    if m and m.group(1) in letters:
        return m.group(1)
    # cuối câu: "... D."
    m = re.search(r"\b([A-Z])\b\.?\s*$", text.strip())
    if m and m.group(1) in letters:
        return m.group(1)
    return None


# ============================================================================
# 2. Few-shot prompt pools
# ============================================================================

GSM8K_5SHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
     "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. After planting there are 21 trees. "
     "So they planted 21 - 15 = 6 trees. The answer is \\boxed{6}."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive, so 3 + 2 = 5. The answer is \\boxed{5}."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Leah had 32 and her sister had 42, so 32 + 42 = 74 in total. "
     "After eating 35, they have 74 - 35 = 39. The answer is \\boxed{39}."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
     "How many lollipops did Jason give to Denny?",
     "Jason started with 20 and now has 12, so he gave 20 - 12 = 8. The answer is \\boxed{8}."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
     "How many toys does he have now?",
     "Shawn started with 5 toys. He got 2 from mom and 2 from dad = 4 more. "
     "Total: 5 + 4 = 9. The answer is \\boxed{9}."),
]

MATH_4SHOT = [
    ("Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
     "We need x - 2 >= 0 and 5 - x > 0, i.e. 2 <= x < 5. "
     "The domain is \\boxed{[2,5)}."),
    ("If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
     "det(AB) = det(A) det(B) = 2 * 12 = 24. The answer is \\boxed{24}."),
    ("Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, "
     "how many times must Terrell lift them in order to lift the same total weight?",
     "Total normally = 2 * 20 * 12 = 480. With 15-lb each lift moves 30 lb. "
     "Need 480 / 30 = 16 lifts. The answer is \\boxed{16}."),
    ("If the system of equations\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution "
     "$(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
     "Multiply first eq by -3/2: -9x + 6y = -3a/2. So b = -3a/2, "
     "i.e. a/b = -2/3. The answer is \\boxed{-\\frac{2}{3}}."),
]


def build_fewshot_block(shots, q_prefix="Question: ", a_prefix="Answer: ") -> str:
    return "\n\n".join(f"{q_prefix}{q}\n{a_prefix}{a}" for q, a in shots)


# ============================================================================
# 3. Model loading
# ============================================================================

def load_model_and_tokenizer(
    model_name: str = "pkcii/distillm2-sft",
    subfolder: str = "checkpoint-1140",
    base_tokenizer: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    device: str = "cuda:1",
):
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = {"torch_dtype": torch.float16, "trust_remote_code": True, "device_map": device}
    if subfolder:
        kwargs["subfolder"] = subfolder
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


# ============================================================================
# 4. Generic batched generation
# ============================================================================

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 784,
    do_sample: bool = False,
    temperature: float = 0.0,
    return_only_completion: bool = True,
    max_input_length: int = 3072,
) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    if return_only_completion:
        out = out[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def chat_prompt(tokenizer, system: Optional[str], user: str) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ============================================================================
# 5. Per-benchmark evaluators
# ============================================================================

# ---------- 5.1 GSM8K (5-shot) ----------

def eval_gsm8k(model, tokenizer, batch_size=64, max_new_tokens=512) -> Dict[str, Any]:
    print("\n" + "=" * 60 + "\nGSM8K (5-shot)\n" + "=" * 60)
    ds = load_dataset("gsm8k", "main", split="test")
    fewshot = build_fewshot_block(GSM8K_5SHOT)
    sys_msg = "You are a math problem solver. Put your final answer within \\boxed{}."

    correct, total = 0, len(ds)
    for i in tqdm(range(0, total, batch_size), desc="GSM8K"):
        batch = ds[i : i + batch_size]
        prompts, golds = [], []
        for q, a in zip(batch["question"], batch["answer"]):
            user = (f"Solve the following problem step by step. "
                    f"Here are some examples:\n\n{fewshot}\n\nQuestion: {q}\nAnswer:")
            prompts.append(chat_prompt(tokenizer, sys_msg, user))
            golds.append(extract_gsm8k_gold(a))

        outs = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        for pred_text, gold in zip(outs, golds):
            pred = extract_boxed(pred_text)
            if math_answers_equal(pred, gold):
                correct += 1

    acc = correct / total
    print(f"GSM8K acc: {acc:.4%}  ({correct}/{total})")
    return {"name": "gsm8k", "n_shot": 5, "acc": acc, "correct": correct, "total": total}


# ---------- 5.2 GSM-Plus (5-shot) ----------

def eval_gsmplus(model, tokenizer, batch_size=64, max_new_tokens=512) -> Dict[str, Any]:
    print("\n" + "=" * 60 + "\nGSM-Plus (5-shot)\n" + "=" * 60)
    ds = load_dataset("qintongli/GSM-Plus", split="test")
    fewshot = build_fewshot_block(GSM8K_5SHOT)
    sys_msg = "You are a math problem solver. Put your final answer within \\boxed{}."

    correct, total = 0, len(ds)
    for i in tqdm(range(0, total, batch_size), desc="GSM-Plus"):
        batch = ds[i : i + batch_size]
        prompts, golds = [], []
        for q, a in zip(batch["question"], batch["answer"]):
            user = (f"Solve the following problem step by step. "
                    f"Here are some examples:\n\n{fewshot}\n\nQuestion: {q}\nAnswer:")
            prompts.append(chat_prompt(tokenizer, sys_msg, user))
            # GSM-Plus có thể cho 'answer' dạng số trực tiếp hoặc có ####
            gold = a if "####" not in str(a) else extract_gsm8k_gold(a)
            golds.append(str(gold).strip())

        outs = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        for pred_text, gold in zip(outs, golds):
            pred = extract_boxed(pred_text)
            if math_answers_equal(pred, gold):
                correct += 1

    acc = correct / total
    print(f"GSM-Plus acc: {acc:.4%}  ({correct}/{total})")
    return {"name": "gsm_plus", "n_shot": 5, "acc": acc, "correct": correct, "total": total}


# ---------- 5.3 MATH-500 (4-shot) ----------

def eval_math500(model, tokenizer, batch_size=32, max_new_tokens=784) -> Dict[str, Any]:
    print("\n" + "=" * 60 + "\nMATH-500 (4-shot)\n" + "=" * 60)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    fewshot = build_fewshot_block(MATH_4SHOT)
    sys_msg = "You are a math expert. Put your final answer within \\boxed{}."

    correct, total = 0, len(ds)
    for i in tqdm(range(0, total, batch_size), desc="MATH-500"):
        batch = ds[i : i + batch_size]
        prompts, golds = [], []
        for prob, sol in zip(batch["problem"], batch["solution"]):
            user = (f"Solve the following problem step by step. "
                    f"Here are some examples:\n\n{fewshot}\n\nQuestion: {prob}\nAnswer:")
            prompts.append(chat_prompt(tokenizer, sys_msg, user))
            golds.append(extract_boxed(sol))

        outs = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        for pred_text, gold in zip(outs, golds):
            pred = extract_boxed(pred_text)
            if math_answers_equal(pred, gold):
                correct += 1

    acc = correct / total
    print(f"MATH-500 acc: {acc:.4%}  ({correct}/{total})")
    return {"name": "math500", "n_shot": 4, "acc": acc, "correct": correct, "total": total}


# ---------- 5.4 MMLU-Pro-Math (5-shot) ----------

def _format_mmlu_pro_question(question, options) -> str:
    letters = string.ascii_uppercase[: len(options)]
    body = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    return f"{question}\n{body}"


def eval_mmlu_pro_math(model, tokenizer, batch_size=16, max_new_tokens=512,
                      n_shot=5) -> Dict[str, Any]:
    print("\n" + "=" * 60 + "\nMMLU-Pro-Math (5-shot)\n" + "=" * 60)
    ds_full = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds_full.filter(lambda x: x["category"].lower() == "math")
    val = load_dataset("TIGER-Lab/MMLU-Pro", split="validation").filter(
        lambda x: x["category"].lower() == "math")

    # build few-shot from validation
    shots = []
    for ex in val.select(range(min(n_shot, len(val)))):
        q = _format_mmlu_pro_question(ex["question"], ex["options"])
        cot = ex.get("cot_content", "") or ""
        cot = cot.replace("A: ", "").strip()
        if not cot:
            cot = f"The answer is \\boxed{{{ex['answer']}}}."
        shots.append((q, cot))
    fewshot = build_fewshot_block(shots, q_prefix="Question: ", a_prefix="Answer: ")

    sys_msg = ("The following are multiple choice questions (with answers). "
               "Think step by step and end with 'The answer is \\boxed{X}'.")

    correct, total = 0, len(ds)
    n_choices_max = 10  # MMLU-Pro có thể đến 10 lựa chọn
    for i in tqdm(range(0, total, batch_size), desc="MMLU-Pro-Math"):
        batch = ds[i : i + batch_size]
        prompts, golds = [], []
        for q, opts, ans in zip(batch["question"], batch["options"], batch["answer"]):
            qtext = _format_mmlu_pro_question(q, opts)
            user = f"{fewshot}\n\nQuestion: {qtext}\nAnswer:"
            prompts.append(chat_prompt(tokenizer, sys_msg, user))
            golds.append(ans.strip().upper())

        outs = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        for pred_text, gold in zip(outs, golds):
            pred = extract_choice_letter(pred_text, n_choices=n_choices_max)
            if pred == gold:
                correct += 1

    acc = correct / total if total else 0.0
    print(f"MMLU-Pro-Math acc: {acc:.4%}  ({correct}/{total})")
    return {"name": "mmlu_pro_math", "n_shot": n_shot, "acc": acc,
            "correct": correct, "total": total}


# ---------- 5.5 MMLU-STEM (5-shot) ----------

MMLU_STEM_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_physics",
    "computer_security", "conceptual_physics", "electrical_engineering",
    "elementary_mathematics", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics", "high_school_physics",
    "high_school_statistics", "machine_learning",
]


def _format_mmlu_question(q, choices) -> str:
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"{l}. {o}" for l, o in zip(letters, choices))
    return f"{q}\n{body}"


def eval_mmlu_stem(model, tokenizer, batch_size=32, max_new_tokens=256,
                   n_shot=5, max_per_subject=None) -> Dict[str, Any]:
    print("\n" + "=" * 60 + "\nMMLU-STEM (5-shot)\n" + "=" * 60)
    sys_msg = ("The following are multiple choice questions (with answers) about "
               "{subject}. Think step by step and end with 'The answer is \\boxed{{X}}'.")

    total_correct, grand_total = 0, 0
    per_subject = {}

    for subj in MMLU_STEM_SUBJECTS:
        try:
            dev = load_dataset("cais/mmlu", subj, split="dev")
            test = load_dataset("cais/mmlu", subj, split="test")
        except Exception as e:
            print(f"  skip {subj}: {e}")
            continue
        if max_per_subject:
            test = test.select(range(min(max_per_subject, len(test))))

        # few-shot từ dev
        shots = []
        for ex in dev.select(range(min(n_shot, len(dev)))):
            qf = _format_mmlu_question(ex["question"], ex["choices"])
            letter = ["A", "B", "C", "D"][ex["answer"]]
            shots.append((qf, f"The answer is \\boxed{{{letter}}}."))
        fewshot = build_fewshot_block(shots)

        subj_pretty = subj.replace("_", " ")
        sys_for_subj = sys_msg.format(subject=subj_pretty)

        s_correct, s_total = 0, len(test)
        for i in tqdm(range(0, s_total, batch_size), desc=f"MMLU/{subj}", leave=False):
            batch = test[i : i + batch_size]
            prompts, golds = [], []
            for q, ch, ans in zip(batch["question"], batch["choices"], batch["answer"]):
                qf = _format_mmlu_question(q, ch)
                user = f"{fewshot}\n\nQuestion: {qf}\nAnswer:"
                prompts.append(chat_prompt(tokenizer, sys_for_subj, user))
                golds.append(["A", "B", "C", "D"][ans])

            outs = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
            for pred_text, gold in zip(outs, golds):
                pred = extract_choice_letter(pred_text, n_choices=4)
                if pred == gold:
                    s_correct += 1

        per_subject[subj] = {"acc": s_correct / s_total, "n": s_total}
        print(f"  {subj}: {s_correct}/{s_total} = {s_correct/s_total:.3%}")
        total_correct += s_correct
        grand_total += s_total

    acc = total_correct / grand_total if grand_total else 0.0
    print(f"MMLU-STEM macro-pool acc: {acc:.4%}  ({total_correct}/{grand_total})")
    return {"name": "mmlu_stem", "n_shot": n_shot, "acc": acc,
            "correct": total_correct, "total": grand_total, "per_subject": per_subject}


# ---------- 5.6 SciQ (0-shot) ----------

def eval_sciq(model, tokenizer, batch_size=32, max_new_tokens=256) -> Dict[str, Any]:
    print("\n" + "=" * 60 + "\nSciQ (0-shot)\n" + "=" * 60)
    ds = load_dataset("allenai/sciq", split="test")
    sys_msg = ("Answer the following science multiple choice question. "
               "Think step by step and end with 'The answer is \\boxed{X}' "
               "where X is one of A, B, C, D.")

    correct, total = 0, len(ds)
    # cố định thứ tự đáp án để reproducible
    import random
    rng = random.Random(42)
    for i in tqdm(range(0, total, batch_size), desc="SciQ"):
        batch = ds[i : i + batch_size]
        prompts, golds = [], []
        for q, ca, d1, d2, d3 in zip(batch["question"], batch["correct_answer"],
                                     batch["distractor1"], batch["distractor2"],
                                     batch["distractor3"]):
            opts = [ca, d1, d2, d3]
            order = list(range(4))
            rng.shuffle(order)
            opts = [opts[k] for k in order]
            gold_letter = "ABCD"[order.index(0)]
            qf = _format_mmlu_question(q, opts)
            prompts.append(chat_prompt(tokenizer, sys_msg, f"Question: {qf}\nAnswer:"))
            golds.append(gold_letter)

        outs = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        for pred_text, gold in zip(outs, golds):
            if extract_choice_letter(pred_text) == gold:
                correct += 1

    acc = correct / total
    print(f"SciQ acc: {acc:.4%}  ({correct}/{total})")
    return {"name": "sciq", "n_shot": 0, "acc": acc, "correct": correct, "total": total}


# ---------- 5.7 MBPP (0-shot, code execution pass@1) ----------

MBPP_PROMPT_TEMPLATE = (
    "You are an expert Python programmer. Write a Python function that solves the task.\n"
    "Return ONLY the function code inside a ```python ... ``` block; no explanation.\n\n"
    "Task: {prompt}\n\n"
    "Your code must pass these tests:\n{tests}\n"
)


def _extract_python_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _run_python_code(code: str, tests: List[str], timeout: int = 8) -> bool:
    """Chạy code trong subprocess với timeout. Trả về True nếu mọi test pass."""
    full = code + "\n" + "\n".join(tests) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        r = subprocess.run(
            ["python", path],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def eval_mbpp(model, tokenizer, batch_size=8, max_new_tokens=512) -> Dict[str, Any]:
    print("\n" + "=" * 60 + "\nMBPP (0-shot, code exec)\n" + "=" * 60)
    # MBPP sanitized: 257 task; full mbpp 'test' split là 500.
    ds = load_dataset("mbpp", "sanitized", split="test")

    sys_msg = "You are a helpful Python programming assistant."
    correct, total = 0, len(ds)
    for i in tqdm(range(0, total, batch_size), desc="MBPP"):
        batch = ds[i : i + batch_size]
        prompts, tests_list = [], []
        for prompt, tests in zip(batch["prompt"], batch["test_list"]):
            user = MBPP_PROMPT_TEMPLATE.format(prompt=prompt, tests="\n".join(tests))
            prompts.append(chat_prompt(tokenizer, sys_msg, user))
            tests_list.append(tests)

        outs = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        for pred_text, tests in zip(outs, tests_list):
            code = _extract_python_code(pred_text)
            if _run_python_code(code, tests):
                correct += 1

    acc = correct / total
    print(f"MBPP pass@1: {acc:.4%}  ({correct}/{total})")
    return {"name": "mbpp", "n_shot": 0, "acc": acc, "correct": correct, "total": total}


# ---------- 5.8 BBH (3-shot) ----------

# Danh sách 27 task BBH chuẩn (Suzgun et al., 2022)
BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]


def _bbh_extract(pred: str) -> str:
    """BBH chính thức dùng pattern 'So the answer is X.'"""
    m = re.search(r"[Ss]o the answer is\s*(.*?)\.", pred)
    if m:
        return m.group(1).strip()
    m = re.search(r"answer is\s*(.*?)\.?\s*$", pred.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # boxed
    b = extract_boxed(pred)
    if b is not None:
        return b
    return pred.strip().split("\n")[-1].strip().rstrip(".")


def _bbh_normalize(s: str) -> str:
    s = s.strip().rstrip(".").strip()
    # lấy nội dung trong (X) -> X
    m = re.match(r"^\(([A-Za-z0-9])\)$", s)
    if m:
        return m.group(1).upper()
    return s.lower()


def eval_bbh(model, tokenizer, batch_size=16, max_new_tokens=512,
             max_per_task=None) -> Dict[str, Any]:
    """
    BBH 3-shot CoT. Dùng prompt CoT 3-shot lấy trực tiếp từ repo BBH (lukaemon/bbh).
    Repo `lukaemon/bbh` chỉ có data; few-shot prompt nằm ở repo gốc của Suzgun
    (https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/cot-prompts).
    Ở đây dataset đã chứa sẵn trường `targets`, ta tự xây 3-shot từ vài ví dụ
    đầu tiên của task để tránh phụ thuộc internet.
    """
    print("\n" + "=" * 60 + "\nBBH (3-shot CoT)\n" + "=" * 60)
    sys_msg = ("Solve the task step by step. End your response with "
               "'So the answer is X.' where X is the final answer.")

    grand_correct, grand_total = 0, 0
    per_task = {}

    for task in BBH_TASKS:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
        except Exception as e:
            print(f"  skip {task}: {e}")
            continue

        # few-shot: dùng 3 example đầu (rồi loại khỏi eval)
        shots_data = ds.select(range(3))
        eval_data = ds.select(range(3, len(ds)))
        if max_per_task:
            eval_data = eval_data.select(range(min(max_per_task, len(eval_data))))

        shots = []
        for ex in shots_data:
            shots.append((ex["input"],
                          f"Let's think step by step.\nSo the answer is {ex['target']}."))
        fewshot = build_fewshot_block(shots, q_prefix="Q: ", a_prefix="A: ")

        s_correct, s_total = 0, len(eval_data)
        for i in tqdm(range(0, s_total, batch_size), desc=f"BBH/{task}", leave=False):
            batch = eval_data[i : i + batch_size]
            prompts, golds = [], []
            for q, t in zip(batch["input"], batch["target"]):
                user = f"{fewshot}\n\nQ: {q}\nA: Let's think step by step."
                prompts.append(chat_prompt(tokenizer, sys_msg, user))
                golds.append(t)

            outs = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
            for pred_text, gold in zip(outs, golds):
                pred = _bbh_extract(pred_text)
                if _bbh_normalize(pred) == _bbh_normalize(gold):
                    s_correct += 1

        per_task[task] = {"acc": s_correct / s_total if s_total else 0.0, "n": s_total}
        print(f"  {task}: {s_correct}/{s_total} = "
              f"{(s_correct/s_total) if s_total else 0:.3%}")
        grand_correct += s_correct
        grand_total += s_total

    acc = grand_correct / grand_total if grand_total else 0.0
    print(f"BBH macro-pool acc: {acc:.4%}  ({grand_correct}/{grand_total})")
    return {"name": "bbh", "n_shot": 3, "acc": acc,
            "correct": grand_correct, "total": grand_total, "per_task": per_task}


# ---------- 5.9 MuSR (0-shot CoT) ----------

MUSR_DOMAINS = ["murder_mysteries", "object_placements", "team_allocation"]


def eval_musr(model, tokenizer, batch_size=8, max_new_tokens=1024) -> Dict[str, Any]:
    print("\n" + "=" * 60 + "\nMuSR (0-shot CoT)\n" + "=" * 60)
    sys_msg = ("Read the narrative carefully and answer the multiple choice question. "
               "Think step by step and end with 'The answer is \\boxed{X}' "
               "where X is the letter of the correct option.")

    grand_correct, grand_total = 0, 0
    per_domain = {}

    for domain in MUSR_DOMAINS:
        try:
            ds = load_dataset("TAUR-Lab/MuSR", split=domain)
        except Exception as e:
            print(f"  skip {domain}: {e}")
            continue

        d_correct, d_total = 0, len(ds)
        for i in tqdm(range(0, d_total, batch_size), desc=f"MuSR/{domain}", leave=False):
            batch = ds[i : i + batch_size]
            prompts, golds = [], []
            for narrative, question, choices, ans_idx in zip(
                batch["narrative"], batch["question"],
                batch["choices"], batch["answer_index"]
            ):
                # MuSR lưu choices là string list dạng "['a', 'b', ...]" trong vài revision
                if isinstance(choices, str):
                    try:
                        choices = json.loads(choices.replace("'", '"'))
                    except Exception:
                        choices = eval(choices)  # last resort
                qf = _format_mmlu_question(question, choices)
                user = f"Narrative:\n{narrative}\n\nQuestion: {qf}\nAnswer:"
                prompts.append(chat_prompt(tokenizer, sys_msg, user))
                golds.append("ABCDEFGH"[int(ans_idx)])

            outs = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
            for pred_text, gold in zip(outs, golds):
                pred = extract_choice_letter(pred_text, n_choices=8)
                if pred == gold:
                    d_correct += 1

        per_domain[domain] = {"acc": d_correct / d_total if d_total else 0.0, "n": d_total}
        print(f"  {domain}: {d_correct}/{d_total} = "
              f"{(d_correct/d_total) if d_total else 0:.3%}")
        grand_correct += d_correct
        grand_total += d_total

    acc = grand_correct / grand_total if grand_total else 0.0
    print(f"MuSR macro-pool acc: {acc:.4%}  ({grand_correct}/{grand_total})")
    return {"name": "musr", "n_shot": 0, "acc": acc,
            "correct": grand_correct, "total": grand_total, "per_domain": per_domain}


# ---------- 5.10 IFEval (0-shot, rule-based subset) ----------

def _ifeval_check(prompt: str, response: str, instr_id: str, kwargs: dict) -> bool:
    """
    Kiểm tra rule-based subset của IFEval. Đây là cài đặt rút gọn vài instruction
    phổ biến nhất; full eval cần repo `instruction_following_eval` của Google.
    """
    text = response

    if instr_id == "length_constraints:number_words":
        # ít nhất / nhiều nhất N từ
        n = kwargs.get("num_words")
        rel = kwargs.get("relation", "at least")
        words = len(text.split())
        if rel == "at least":
            return words >= n
        if rel == "less than":
            return words < n
        return False

    if instr_id == "length_constraints:number_sentences":
        n = kwargs.get("num_sentences")
        rel = kwargs.get("relation", "at least")
        sents = re.split(r"[.!?]+", text.strip())
        sents = [s for s in sents if s.strip()]
        if rel == "at least":
            return len(sents) >= n
        if rel == "less than":
            return len(sents) < n

    if instr_id == "change_case:english_lowercase":
        return text == text.lower()

    if instr_id == "change_case:english_capital":
        return text == text.upper()

    if instr_id == "punctuation:no_comma":
        return "," not in text

    if instr_id == "keywords:existence":
        kws = kwargs.get("keywords", [])
        return all(k.lower() in text.lower() for k in kws)

    if instr_id == "keywords:forbidden_words":
        forb = kwargs.get("forbidden_words", [])
        return not any(w.lower() in text.lower() for w in forb)

    if instr_id == "startend:end_checker":
        end = kwargs.get("end_phrase", "")
        return text.strip().endswith(end)

    if instr_id == "format:title":
        return bool(re.search(r"<<.+?>>", text))

    if instr_id == "format:number_bullet_lists":
        n = kwargs.get("num_bullets")
        bullets = re.findall(r"^\s*[\*\-]\s+", text, flags=re.MULTILINE)
        return len(bullets) == n

    # instruction không support -> coi như không đánh giá (không cộng điểm)
    return None


def eval_ifeval(model, tokenizer, batch_size=8, max_new_tokens=1024) -> Dict[str, Any]:
    print("\n" + "=" * 60 + "\nIFEval (0-shot, rule subset)\n" + "=" * 60)
    ds = load_dataset("google/IFEval", split="train")

    n = len(ds)
    instr_total, instr_pass = 0, 0
    prompt_total, prompt_pass = 0, 0
    skipped_instr = 0

    for i in tqdm(range(0, n, batch_size), desc="IFEval"):
        batch = ds[i : i + batch_size]
        prompts = [chat_prompt(tokenizer, None, p) for p in batch["prompt"]]
        outs = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)

        for resp, ids, kw_list, prompt_text in zip(
            outs, batch["instruction_id_list"], batch["kwargs"], batch["prompt"]
        ):
            all_pass = True
            n_evaluated = 0
            for iid, kw in zip(ids, kw_list):
                ok = _ifeval_check(prompt_text, resp, iid, kw or {})
                if ok is None:
                    skipped_instr += 1
                    continue
                instr_total += 1
                n_evaluated += 1
                if ok:
                    instr_pass += 1
                else:
                    all_pass = False
            if n_evaluated > 0:
                prompt_total += 1
                if all_pass:
                    prompt_pass += 1

    instr_acc = instr_pass / instr_total if instr_total else 0.0
    prompt_acc = prompt_pass / prompt_total if prompt_total else 0.0
    print(f"IFEval (subset): instr-level={instr_acc:.4%}  prompt-level={prompt_acc:.4%}")
    print(f"   ({instr_pass}/{instr_total} instr, {prompt_pass}/{prompt_total} prompt, "
          f"{skipped_instr} unsupported instructions skipped)")
    print("   NOTE: chạy full IFEval với `pip install instruction-following-eval` để đầy đủ.")
    return {
        "name": "ifeval", "n_shot": 0,
        "instr_acc": instr_acc, "prompt_acc": prompt_acc,
        "instr_pass": instr_pass, "instr_total": instr_total,
        "prompt_pass": prompt_pass, "prompt_total": prompt_total,
        "skipped_instructions": skipped_instr,
    }


# ============================================================================
# 6. Main runner
# ============================================================================

REGISTRY = {
    "gsm8k":         eval_gsm8k,
    "gsm_plus":      eval_gsmplus,
    "math":          eval_math500,
    "math500":       eval_math500,
    "mmlu_pro_math": eval_mmlu_pro_math,
    "mmlu_stem":     eval_mmlu_stem,
    "sciq":          eval_sciq,
    "mbpp":          eval_mbpp,
    "bbh":           eval_bbh,
    "musr":          eval_musr,
    "ifeval":        eval_ifeval,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pkcii/distillm2-sft")
    parser.add_argument("--subfolder", type=str, default="checkpoint-1140")
    parser.add_argument("--base_tokenizer", type=str,
                        default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--benchmarks", type=str, default="all",
                        help="comma-separated list, hoặc 'all'")
    parser.add_argument("--out", type=str, default="results.json")
    args = parser.parse_args()

    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        subfolder=args.subfolder,
        base_tokenizer=args.base_tokenizer,
        device=args.device,
    )

    if args.benchmarks == "all":
        bench_list = list(REGISTRY.keys())
        # bỏ alias 'math500' khỏi run-all để không chạy MATH 2 lần
        bench_list = [b for b in bench_list if b != "math500"]
    else:
        bench_list = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    results = {}
    for b in bench_list:
        if b not in REGISTRY:
            print(f"[!] benchmark '{b}' không có trong REGISTRY, bỏ qua.")
            continue
        try:
            res = REGISTRY[b](model, tokenizer, batch_size=args.batch_size)
            results[b] = res
        except Exception as e:
            print(f"[!] error evaluating {b}: {e}")
            results[b] = {"error": str(e)}
        # giải phóng cache giữa các benchmark
        gc.collect()
        torch.cuda.empty_cache()

        # ghi kết quả tăng dần
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60 + "\nFINAL\n" + "=" * 60)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()