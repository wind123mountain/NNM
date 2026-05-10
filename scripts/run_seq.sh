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
bash ./scripts/train/qwen2.5-0.5B/sft_qwen2.5_0.5b.sh 2>&1 | tee outputs/logs/sft_qwen2.5_0.5b.log
bash ./scripts/train/qwen2.5-0.5B/distillm_2_qwen2.5_0.5b.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_0.5b.log
bash ./scripts/train/qwen2.5-0.5B/distillm_2_qwen2.5_0.5b-1e.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_0.5b-1e.log

bash /media/volume/ElasticVol/LLM_Distillation/test.sh