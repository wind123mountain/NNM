N_GPUS=${1:-1}

# bash ./scripts/gen_traces/Qwen2.5-Math-1.5B-Instruct.sh $N_GPUS
# bash ./scripts/gen_traces/DeepSeek-R1-Distill-Llama-8B.sh $N_GPUS
bash ./scripts/gen_traces/Llama-3.2-3B-Instruct.sh $N_GPUS
# bash ./scripts/gen_traces/Qwen2.5-0.5B.sh $N_GPUS
