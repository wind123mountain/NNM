import os
import json
import gc
import torch
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

def run_vllm_inference(model_path, output_dir, output_file, tensor_parallel_size=1):
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU CHẠY MODEL: {model_path}")
    print(f"📂 Thư mục lưu: {os.path.join(output_dir, output_file)}")
    print(f"⚙️  Tensor Parallel Size: {tensor_parallel_size}")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)

    print("⏳ Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    data = load_dataset('VoCuc/MetaMathQA-50k-256', split='train')['query']

    print("🧩 Applying chat template...")
    prompts_all = []
    for prompt in data:
        conversation = [
            {"role": "system", "content": "You are a teacher. Solve the problem and put your final answer within \\boxed{}."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts_all.append(formatted_prompt)

    print("🔥 Initializing vLLM...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,
        seed=42
    )

    # ==========================================
    # 3. CHẠY INFERENCE
    # ==========================================
    sampling_params = SamplingParams(
        temperature=0.85,
        top_p=0.95,
        max_tokens=768,
        skip_special_tokens=True
    )

    print(f"⚙️ Generating responses for {len(prompts_all)} prompts...")
    outputs = llm.generate(prompts_all, sampling_params)

    # ==========================================
    # 4. LƯU KẾT QUẢ
    # ==========================================
    output_data = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        output_data.append({
            'prompt': data[i],
            'generated_text': generated_text,
        })

    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Đã lưu xong file tại: {output_path}")

    print("🧹 Đang dọn dẹp bộ nhớ GPU...")
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("✨ Quá trình inference hoàn tất!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM Inference")
    
    # Khai báo các tham số đầu vào
    parser.add_argument("--model_path", type=str, required=True, help="Path or HuggingFace ID of the model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output file")
    parser.add_argument("--output_file", type=str, default="generated_train.jsonl", help="Output file name")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    
    args = parser.parse_args()
    
    run_vllm_inference(
        model_path=args.model_path,
        output_dir=args.output_dir,
        output_file=args.output_file,
        tensor_parallel_size=args.tensor_parallel_size
    )