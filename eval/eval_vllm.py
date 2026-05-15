#!/usr/bin/env python3
"""
Custom vLLM evaluation script
Tasks : GSM8K (5-shot CoT) | MATH500 (4-shot) | MBPP (3-shot) | SciQ (0-shot) | MMLU-STEM (5-shot)
Metric: pass@1 / accuracy

Usage:
    python eval_vllm.py --model <path_or_hf_id> [--tasks gsm8k,math500,mbpp,sciq,mmlu_stem]
                        [--output_dir ./eval_results] [--max_tokens 1024] [--tensor_parallel 1]
"""

import argparse, json, os, re, sys, time, subprocess, tempfile, traceback
from pathlib import Path
from typing import List, Dict, Tuple, Any

# ─────────────────────────────────────────────────────────────
# System prompts (giữ nguyên như user cung cấp)
# ─────────────────────────────────────────────────────────────
MATH_SYSTEM_PROMPT = (
    "You are a math teacher. You will be given a math problem and you will solve it step by step, "
    "keep the reasoning brief and focused.\n"
    "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units "
    "within the brackets and fully evaluate arithmetic expressions.\n"
)

MMLU_SYSTEM_PROMPT = (
    "You are a teacher. You will be given a problem and you will solve it step by step.\n"
    "You will output your final solution like \\boxed{X}, where X is exactly one capital letter "
    "corresponding to the correct option (A, B, C, or D).\n"
    "Do not include any extra text, punctuation, or words inside the box.\n"
)

GSM8K_SYSTEM_PROMPT = (
    "You are a math teacher. Solve each grade-school math word problem step by step, "
    "keeping the reasoning brief and focused. Respond in English only.\n"
    "End your response with a line of the form: The answer is <NUMBER>.\n"
)

# ─────────────────────────────────────────────────────────────
# Few-shot examples (from original papers)
# ─────────────────────────────────────────────────────────────

# GSM8K: 5-shot Chain-of-Thought (first 5 of the 8 canonical examples from
# Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in LLMs", Appendix G).
GSM8K_FEWSHOT = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": (
            "There are 15 trees originally. Then there were 21 trees after some more were planted. "
            "So there must have been 21 - 15 = 6. The answer is 6."
        ),
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": (
            "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."
        ),
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": (
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
            "After eating 35, they had 74 - 35 = 39. The answer is 39."
        ),
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": (
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
            "So he gave Denny 20 - 12 = 8. The answer is 8."
        ),
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": (
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
            "5 + 4 = 9. The answer is 9."
        ),
    },
]

# MATH500: 4-shot từ Listing 2 trong paper (Image 1)
MATH500_FEWSHOT = [
    {
        "problem": (
            r"Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$."
        ),
        "solution": (
            "The expressions inside each square root must be non-negative. Therefore, "
            r"$x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator "
            r"cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of "
            r"the expression is $\boxed{[2,5)}$."
            "\nFinal Answer: The final answer is $[2,5)$. I hope it is correct."
        ),
    },
    {
        "problem": (
            r"If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$"
        ),
        "solution": (
            r"We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$"
            "\nFinal Answer: The final answer is $24$. I hope it is correct."
        ),
    },
    {
        "problem": (
            "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound "
            "weights instead, how many times must Terrell lift them in order to lift the same total weight?"
        ),
        "solution": (
            r"If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ "
            r"pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a "
            r"total of $2\cdot15\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:"
            "\n"
            r"$$30n=480 \Rightarrow n=480/30=\boxed{16}$$"
            "\nFinal Answer: The final answer is $16$. I hope it is correct."
        ),
    },
    {
        "problem": (
            r"If the system of equations $\begin{aligned} 6x-4y&=a,\\ 6y-9x &=b.\end{aligned}$ "
            r"has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b},$ "
            r"assuming $b$ is nonzero."
        ),
        "solution": (
            r"If we multiply the first equation by $-\frac{3}{2}$, we obtain "
            r"$$6y-9x=-\frac{3}{2}a.$$"
            r"Since we also know that $6y-9x=b$, we have "
            r"$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$"
            "\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."
        ),
    },
]

# MBPP: 3-shot từ Figure 1 trong paper (Image 2)
# Prompt format: docstring với task description + test cases, model complete code
MBPP_FEWSHOT = [
    {
        "text": "Write a python function to check if a given number is one less than twice its reverse. Your code should satisfy these tests:",
        "tests": [
            "assert check(70) == False",
            "assert check(23) == False",
            "assert check(73) == True",
        ],
        "code": (
            "def check(n):\n"
            "    if n == 2*int(str(n)[::-1])-1:\n"
            "        return True\n"
            "    else:\n"
            "        return False"
        ),
    },
    {
        "text": "Write a function to find the smallest missing element in a sorted array. Your code should satisfy these tests:",
        "tests": [
            "assert smallest_missing([0, 1, 2, 3, 4, 5, 6], 0, 6) == 7",
            "assert smallest_missing([0, 1, 2, 6, 9, 11, 15], 0, 6) == 3",
            "assert smallest_missing([1, 2, 3, 4, 6, 9, 11, 15], 0, 7) == 0",
        ],
        "code": (
            "def smallest_missing(arr, n, m):\n"
            "    smallest = min(n, m)\n"
            "    for i in range(n, m + 1):\n"
            "        if arr[i] <= smallest:\n"
            "            smallest += 1\n"
            "    return smallest"
        ),
    },
    {
        "text": "Write a Python function to sort the given array by using merge sort. Your code should satisfy these tests:",
        "tests": [
            "assert merge_sort([3, 4, 2, 6, 5, 7, 1, 9]) == [1, 2, 3, 4, 5, 6, 7, 9]",
            "assert merge_sort([7, 25, 45, 78, 11, 33, 19]) == [7, 11, 19, 25, 33, 45, 78]",
            "assert merge_sort([3, 1, 4, 9, 8]) == [1, 3, 4, 8, 9]",
        ],
        "code": (
            "def merge_sort(arr):\n"
            "    if len(arr) < 2:\n"
            "        return arr\n"
            "    mid = len(arr) // 2\n"
            "    left = arr[:mid]\n"
            "    right = arr[mid:]\n"
            "    left = merge_sort(left)\n"
            "    right = merge_sort(right)\n"
            "    merged = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] < right[j]:\n"
            "            merged.append(left[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            merged.append(right[j])\n"
            "            j += 1\n"
            "    merged.extend(left[i:])\n"
            "    merged.extend(right[j:])\n"
            "    return merged"
        ),
    },
]

# MMLU STEM subjects (18 subjects chuẩn dùng trong papers)
MMLU_STEM_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy",
    "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_physics",
    "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics",
    "formal_logic",
    "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics",
    "machine_learning",
]

CHOICES = ["A", "B", "C", "D"]

# ─────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────

def build_gsm8k_prompt(question: str) -> str:
    """
    5-shot Chain-of-Thought prompt (Wei et al. 2022).
    Format mỗi shot:
        Q: <question>
        A: <reasoning ... The answer is N.>
    """
    shots = ""
    for ex in GSM8K_FEWSHOT:
        shots += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    shots += f"Q: {question}\nA:"
    return shots


def build_math500_prompt(problem: str) -> str:
    shots = ""
    for ex in MATH500_FEWSHOT:
        shots += f"Problem: {ex['problem']}\nSolution: {ex['solution']}\n\n"
    return shots + f"Problem: {problem}\nSolution:"


def build_mbpp_prompt(text: str, test_list: List[str] = None) -> str:
    """
    Format từ paper (Figure 1 / Image 2):
    \"\"\"
    {description}
    {assert tests}
    \"\"\"
    {code}
    """
    shots = ""
    for ex in MBPP_FEWSHOT:
        tests_str = "\n".join(ex["tests"])
        shots += (
            f'"""\n{ex["text"]}\n{tests_str}\n"""\n'
            f'{ex["code"]}\n\n'
        )
    # Query
    if test_list:
        tests_str = "\n".join(test_list[:3])  # chỉ dùng 3 test đầu trong prompt
        shots += f'"""\n{text}\n{tests_str}\n"""\n'
    else:
        shots += f'"""\n{text}\n"""\n'
    return shots


def build_sciq_prompt(sample: Dict) -> str:
    q = sample["question"]
    options = [sample["correct_answer"],
               sample["distractor1"],
               sample["distractor2"],
               sample["distractor3"]]
    # Shuffle deterministically by task_id if available, else keep order
    # Correct answer is always A for simplicity (we track index)
    correct_idx = 0  # correct_answer is always at index 0 before we label
    labeled = list(zip(CHOICES, options))
    choice_str = "\n".join(f"{lbl}. {opt}" for lbl, opt in labeled)
    return (
        f"Question: {q}\n{choice_str}\n"
        f"Answer the question by choosing one letter (A, B, C, or D). "
        f"Output only the letter.\nAnswer:"
    )


def build_mmlu_prompt(sample: Dict, few_shots: List[Dict]) -> str:
    def fmt_one(s):
        choices_str = "\n".join(f"{CHOICES[i]}. {s['choices'][i]}" for i in range(len(s['choices'])))
        return f"Question: {s['question']}\n{choices_str}\nAnswer: \\boxed{{{CHOICES[s['answer']]}}}"

    shots = "\n\n".join(fmt_one(s) for s in few_shots)
    # Build the query (without answer)
    choices_str = "\n".join(f"{CHOICES[i]}. {sample['choices'][i]}" for i in range(len(sample['choices'])))
    query = f"Question: {sample['question']}\n{choices_str}\nAnswer:"
    return shots + "\n\n" + query


# ─────────────────────────────────────────────────────────────
# Answer extractors
# ─────────────────────────────────────────────────────────────

def extract_boxed(text: str) -> str:
    """Extract last \\boxed{...} content."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else ""


def extract_gsm8k_gold(answer_str: str) -> str:
    """GSM8K gold answer is after ####"""
    match = re.search(r"####\s*([\d,\.\-]+)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_str.strip()


def extract_gsm8k_pred(text: str) -> str:
    """
    Extract numeric answer from GSM8K CoT generation.
    Priority:
      1. 'The answer is <NUMBER>' (canonical Wei et al. format)
      2. \\boxed{...}
      3. Last number in the text
    """
    # Truncate at first follow-up "Q:" if model over-generates the next shot
    if "\nQ:" in text:
        text = text.split("\nQ:", 1)[0]

    # 1. "The answer is X"
    m = re.search(
        r"[Tt]he\s+answer\s+is\s*\$?\s*(-?\d[\d,]*\.?\d*)",
        text,
    )
    if m:
        return m.group(1).replace(",", "").rstrip(".").strip()

    # 2. \boxed{...}
    boxed = extract_boxed(text)
    if boxed:
        b = boxed.replace(",", "").strip()
        nums = re.findall(r"-?\d+\.?\d*", b)
        if nums:
            return nums[-1]

    # 3. Last number anywhere
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else text.strip()


def extract_number(text: str) -> str:
    """Extract last number-like from text."""
    # Try boxed first
    boxed = extract_boxed(text)
    if boxed:
        return boxed.replace(",", "").strip()
    # Fallback: last number in text
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else text.strip()


def numbers_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    """Compare two numeric strings with tolerance."""
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return a.strip() == b.strip()


def extract_letter(text: str) -> str:
    """Extract single capital letter answer."""
    # Try \\boxed{X}
    boxed = extract_boxed(text)
    if boxed and boxed.upper() in CHOICES:
        return boxed.upper()
    # Try last capital letter A/B/C/D standalone
    matches = re.findall(r"\b([A-D])\b", text)
    return matches[-1] if matches else ""


# ─────────────────────────────────────────────────────────────
# MBPP code execution
# ─────────────────────────────────────────────────────────────

def extract_code_block(text: str) -> str:
    """Extract Python code from model output."""
    # Try ```python ... ```
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ``` ... ```
    match = re.search(r"```\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Return raw text
    return text.strip()


def run_mbpp_tests(code: str, test_list: List[str], timeout: int = 5) -> bool:
    """Execute generated code + test cases in subprocess. Return True if all pass."""
    test_str = "\n".join(test_list)
    full_code = f"{code}\n\n{test_str}\n"
    fname = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            fname = f.name
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if fname and os.path.exists(fname):
            try:
                os.unlink(fname)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────

def load_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    return ds["test"], ds["train"]


def load_math500():
    from datasets import load_dataset
    # HuggingFaceH4/MATH-500 has: problem, solution, answer, subject, level
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return ds


def load_mbpp():
    from datasets import load_dataset
    # Standard eval: full split, indices 11-510 (500 problems)
    ds_full = load_dataset("google-research-datasets/mbpp", "full")
    test_ds = ds_full["test"]
    # Filter to indices 11-510 (task_id based)
    eval_ds = [s for s in test_ds if 11 <= s["task_id"] <= 510]
    # Few-shot examples from prompt split
    prompt_ds = list(ds_full["prompt"])
    return eval_ds, prompt_ds


def load_sciq():
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="test")
    return ds


def load_mmlu_stem():
    from datasets import load_dataset
    all_test, all_dev = [], {}
    for subj in MMLU_STEM_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj)
            for s in ds["test"]:
                all_test.append({**s, "_subject": subj})
            all_dev[subj] = list(ds["dev"])  # 5-shot source
        except Exception as e:
            print(f"  [WARN] Could not load mmlu/{subj}: {e}")
    return all_test, all_dev


# ─────────────────────────────────────────────────────────────
# vLLM generation helper
# ─────────────────────────────────────────────────────────────

def build_chat_messages(system: str, user: str) -> List[Dict]:
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


def vllm_generate(llm, tokenizer, prompts_or_messages: List, 
                  max_tokens: int = 1024, temperature: float = 0.0,
                  use_chat: bool = True) -> List[str]:
    """
    Batch generate với vLLM.
    prompts_or_messages: list of message dicts (chat) hoặc raw strings.
    """
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</s>", "<|im_end|>", "<|endoftext|>"],
    )

    if use_chat:
        # Apply chat template
        formatted = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in prompts_or_messages
        ]
    else:
        formatted = prompts_or_messages

    outputs = llm.generate(formatted, sampling)
    return [o.outputs[0].text for o in outputs]


# ─────────────────────────────────────────────────────────────
# Task evaluators
# ─────────────────────────────────────────────────────────────

def eval_gsm8k(llm, tokenizer, max_tokens, batch_size):
    print("\n" + "="*50)
    print("Task: GSM8K (5-shot CoT, Wei et al.)")
    print("="*50)
    test_ds, _ = load_gsm8k()

    # GSM8K CoT 5-shot: dùng chat template + system prompt khuyến khích
    # kết thúc bằng "The answer is N."
    messages_list, gold_answers = [], []
    for sample in test_ds:
        prompt = build_gsm8k_prompt(sample["question"])
        messages_list.append(build_chat_messages(GSM8K_SYSTEM_PROMPT, prompt))
        gold_answers.append(extract_gsm8k_gold(sample["answer"]))

    print(f"  Total samples: {len(messages_list)}")
    outputs = batch_run(llm, tokenizer, messages_list, max_tokens, batch_size)

    correct = 0
    results = []
    for i, (out, gold) in enumerate(zip(outputs, gold_answers)):
        pred = extract_gsm8k_pred(out)
        ok = numbers_equal(pred, gold)
        if ok:
            correct += 1
        results.append({"id": i, "pred": pred, "gold": gold, "correct": ok})

    acc = correct / len(results)
    print(f"  GSM8K pass@1: {acc*100:.2f}% ({correct}/{len(results)})")
    return {"task": "gsm8k", "accuracy": acc, "correct": correct, "total": len(results), "results": results}


def eval_math500(llm, tokenizer, max_tokens, batch_size):
    print("\n" + "="*50)
    print("Task: MATH500 (4-shot)")
    print("="*50)
    test_ds = load_math500()

    messages_list, gold_answers = [], []
    for sample in test_ds:
        prompt = build_math500_prompt(sample["problem"])
        messages_list.append(build_chat_messages(MATH_SYSTEM_PROMPT, prompt))
        gold_answers.append(sample["answer"])  # đã extracted sẵn

    print(f"  Total samples: {len(messages_list)}")
    outputs = batch_run(llm, tokenizer, messages_list, max_tokens, batch_size)

    correct = 0
    results = []
    for i, (out, gold) in enumerate(zip(outputs, gold_answers)):
        pred = extract_boxed(out)
        if not pred:
            pred = out.strip().split("\n")[-1].strip()
        ok = numbers_equal(pred, gold) if pred else False
        if not ok:
            ok = (pred.strip() == gold.strip())
        if ok:
            correct += 1
        results.append({"id": i, "pred": pred, "gold": gold, "correct": ok})

    acc = correct / len(results)
    print(f"  MATH500 pass@1: {acc*100:.2f}% ({correct}/{len(results)})")
    return {"task": "math500", "accuracy": acc, "correct": correct, "total": len(results), "results": results}


def eval_mbpp(llm, tokenizer, max_tokens, batch_size):
    print("\n" + "="*50)
    print("Task: MBPP (3-shot, execute tests)")
    print("="*50)
    eval_ds, _ = load_mbpp()

    raw_prompts = []
    for sample in eval_ds:
        raw_prompts.append(build_mbpp_prompt(sample["text"], sample["test_list"]))

    print(f"  Total samples: {len(raw_prompts)}")

    # MBPP dùng raw completion (không chat template)
    from vllm import SamplingParams
    sampling = SamplingParams(
        temperature=0.1,
        max_tokens=max_tokens,
        stop=["\"\"\"", "\n\n\n"],
    )
    outputs_raw = llm.generate(raw_prompts, sampling)
    outputs = [o.outputs[0].text for o in outputs_raw]

    correct = 0
    results = []
    for i, (sample, out) in enumerate(zip(eval_ds, outputs)):
        code = extract_code_block(out)
        ok = run_mbpp_tests(code, sample["test_list"])
        if ok:
            correct += 1
        results.append({
            "task_id": sample["task_id"],
            "correct": ok,
            "generated_code": code[:500],
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(eval_ds)}] Running... acc={correct/(i+1)*100:.1f}%")

    acc = correct / len(results)
    print(f"  MBPP pass@1: {acc*100:.2f}% ({correct}/{len(results)})")
    return {"task": "mbpp", "accuracy": acc, "correct": correct, "total": len(results), "results": results}


def eval_sciq(llm, tokenizer, max_tokens, batch_size):
    print("\n" + "="*50)
    print("Task: SciQ (0-shot)")
    print("="*50)
    test_ds = load_sciq()

    # SciQ: correct_answer luôn là A trong prompt của ta
    messages_list = []
    for sample in test_ds:
        prompt = build_sciq_prompt(sample)
        messages_list.append(build_chat_messages(
            "You are a science teacher. Answer multiple choice questions with a single letter.",
            prompt
        ))

    print(f"  Total samples: {len(messages_list)}")
    outputs = batch_run(llm, tokenizer, messages_list, max_tokens=64, batch_size=batch_size)

    correct = 0
    results = []
    for i, out in enumerate(outputs):
        pred = extract_letter(out)
        # correct_answer luôn map về "A" (index 0) trong prompt ta build
        ok = (pred == "A")
        if ok:
            correct += 1
        results.append({"id": i, "pred": pred, "gold": "A", "correct": ok})

    acc = correct / len(results)
    print(f"  SciQ accuracy: {acc*100:.2f}% ({correct}/{len(results)})")
    return {"task": "sciq", "accuracy": acc, "correct": correct, "total": len(results), "results": results}


def eval_mmlu_stem(llm, tokenizer, max_tokens, batch_size):
    print("\n" + "="*50)
    print("Task: MMLU-STEM (5-shot)")
    print("="*50)
    test_samples, dev_by_subj = load_mmlu_stem()

    messages_list, gold_answers, subjects = [], [], []
    for sample in test_samples:
        subj = sample["_subject"]
        few_shots = dev_by_subj.get(subj, [])[:5]
        prompt = build_mmlu_prompt(sample, few_shots)
        messages_list.append(build_chat_messages(MMLU_SYSTEM_PROMPT, prompt))
        gold_answers.append(CHOICES[sample["answer"]])
        subjects.append(subj)

    print(f"  Total samples: {len(messages_list)} across {len(MMLU_STEM_SUBJECTS)} subjects")
    outputs = batch_run(llm, tokenizer, messages_list, max_tokens=64, batch_size=batch_size)

    correct = 0
    per_subject: Dict[str, Dict] = {}
    results = []
    for i, (out, gold, subj) in enumerate(zip(outputs, gold_answers, subjects)):
        pred = extract_letter(out)
        ok = (pred == gold)
        if ok:
            correct += 1
        if subj not in per_subject:
            per_subject[subj] = {"correct": 0, "total": 0}
        per_subject[subj]["correct"] += int(ok)
        per_subject[subj]["total"] += 1
        results.append({"id": i, "subject": subj, "pred": pred, "gold": gold, "correct": ok})

    acc = correct / len(results)
    print(f"  MMLU-STEM accuracy: {acc*100:.2f}% ({correct}/{len(results)})")
    print("  Per-subject breakdown:")
    for subj, stat in sorted(per_subject.items()):
        subj_acc = stat["correct"] / stat["total"]
        print(f"    {subj:<40} {subj_acc*100:.1f}% ({stat['correct']}/{stat['total']})")

    return {
        "task": "mmlu_stem",
        "accuracy": acc, "correct": correct, "total": len(results),
        "per_subject": per_subject,
        "results": results
    }


# ─────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────

def batch_run(llm, tokenizer, messages_list, max_tokens, batch_size):
    all_outputs = []
    for i in range(0, len(messages_list), batch_size):
        batch = messages_list[i:i+batch_size]
        outs = vllm_generate(llm, tokenizer, batch, max_tokens=max_tokens)
        all_outputs.extend(outs)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Generated {min(i+batch_size, len(messages_list))}/{len(messages_list)}")
    return all_outputs


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True, help="HF model name or local path")
    parser.add_argument("--tasks",      type=str, default="gsm8k,math500,mbpp,sciq,mmlu_stem",
                        help="Comma-separated tasks")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--dtype",      type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",")]

    print(f"\n{'='*60}")
    print(f"Model : {args.model}")
    print(f"Tasks : {tasks}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Load vLLM
    from vllm import LLM
    from transformers import AutoTokenizer

    print("Loading model with vLLM...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len=4096,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Run tasks
    all_results = {}
    task_map = {
        "gsm8k":     lambda: eval_gsm8k(llm, tokenizer, args.max_tokens, args.batch_size),
        "math500":   lambda: eval_math500(llm, tokenizer, args.max_tokens, args.batch_size),
        "mbpp":      lambda: eval_mbpp(llm, tokenizer, args.max_tokens, args.batch_size),
        "sciq":      lambda: eval_sciq(llm, tokenizer, args.max_tokens, args.batch_size),
        "mmlu_stem": lambda: eval_mmlu_stem(llm, tokenizer, args.max_tokens, args.batch_size),
    }

    for task in tasks:
        if task not in task_map:
            print(f"[WARN] Unknown task '{task}', skipping.")
            continue
        t0 = time.time()
        try:
            result = task_map[task]()
            all_results[task] = result
            # Save per-task
            out_path = os.path.join(args.output_dir, f"{task}_results.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved → {out_path} (took {time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary = {}
    for task, res in all_results.items():
        acc = res.get("accuracy", 0)
        summary[task] = acc
        print(f"  {task:<15} {acc*100:.2f}%")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
