GPUS=(0 1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
N_GPUS=${#GPUS[@]}

mkdir -p outputs/logs
# mkdir -p /media/volume/ElasticVol/hf_cache
# export HF_HOME=/media/volume/ElasticVol/hf_cache
# export HUGGINGFACE_HUB_CACHE=/media/volume/ElasticVol/hf_cache
# export TRANSFORMERS_CACHE=/media/volume/ElasticVol/hf_cache
# export HF_DATASETS_CACHE=/media/volume/ElasticVol/hf_cache

# bash ./scripts/gen_traces/gen_all.sh $N_GPUS

# bash ./scripts/train/qwen2.5-0.5B/reformat_data.sh
# bash ./scripts/train/qwen2.5-0.5B/sft.sh 2>&1 | tee outputs/logs/sft_qwen2.5_0.5b.log
# bash ./scripts/train/qwen2.5-0.5B/distillm_2.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_0.5b.log
# bash ./scripts/train/qwen2.5-0.5B/distillm_2_1e.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_0.5b-1e.log

if [ ! -d "data/reformatted/distill-deepSeek-R1-Distill-Llama-8B" ]; then
    bash ./scripts/train/llama3.2-3B-it/reformat_data.sh
fi
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/train/llama3.2-3B-it/sft.sh 2>&1 | tee outputs/logs/llama3.2_3b_it_sft.log &
CUDA_VISIBLE_DEVICES=2,3 bash ./scripts/train/llama3.2-3B-it/distillm_2.sh 2>&1 | tee outputs/logs/llama3.2_3b_it_distillm_2.log &
# CUDA_VISIBLE_DEVICES=4,5 bash ./scripts/train/llama3.2-3B-it/distillm_2_1e.sh 2>&1 | tee outputs/logs/llama3.2_3b_it_distillm_2_1e.log &


