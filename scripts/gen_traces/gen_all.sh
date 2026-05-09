N_GPUS=${1:-2}

mkdir -p logs

bash ./scripts/gen_traces/Qwen2.5-Math-7B-Instruct.sh $N_GPUS
