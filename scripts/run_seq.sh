# GPUS=(0 1)
# export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
N_GPUS=8

mkdir -p outputs/logs
# mkdir -p /media/volume/ElasticVol/hf_cache
# export HF_HOME=/media/volume/ElasticVol/hf_cache
# export HUGGINGFACE_HUB_CACHE=/media/volume/ElasticVol/hf_cache
# export TRANSFORMERS_CACHE=/media/volume/ElasticVol/hf_cache
# export HF_DATASETS_CACHE=/media/volume/ElasticVol/hf_cache

# bash ./scripts/gen_traces/gen_all.sh $N_GPUS


if [ ! -d "data/reformatted/distill-qwen2.5-Math-1.5B-Instruct" ]; then
    bash ./scripts/train/qwen2.5-0.5B/reformat_data.sh
fi

if [ ! -d "data/reformatted/distill-deepSeek-R1-Distill-Llama-8B" ]; then
    bash ./scripts/train/llama3.2-3B-it/reformat_data.sh
fi

if [ ! -d "data/reformatted/distill-qwen2.5-14B-Instruct" ]; then
    bash ./scripts/train/qwen2.5-1.5B-it/reformat_data.sh
fi

# (
#     # echo "[GPU 0,1] Start SFT Qwen 0.5B"
#     # CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/train/qwen2.5-0.5B/sft.sh 2>&1 | tee outputs/logs/sft_qwen2.5_0.5b.log
    
#     # echo "[GPU 0,1] Start SFT Llama 3.2 3B"
#     # CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/train/llama3.2-3B-it/sft.sh 2>&1 | tee outputs/logs/llama3.2_3b_it_sft.log

#     echo "[GPU 0,1] Start Distill-1e Qwen 0.5B"
#     CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/train/qwen2.5-0.5B/distillm_2_1e.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_0.5b-1e.log
    
#     echo "[GPU 0,1] Start Distill Qwen 0.5B"
#     CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/train/qwen2.5-0.5B/distillm_2.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_0.5b.log
    
#     echo "[GPU 0,1] Start SFT Qwen 1.5B"
#     CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/train/qwen2.5-1.5B-it/sft.sh 2>&1 | tee outputs/logs/sft_qwen2.5_1.5b_it.log

#     echo "[GPU 0,1] Start Distill-1e Qwen 1.5B"
#     CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/train/qwen2.5-1.5B-it/distillm_2_1e.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_1.5b_1e.log
    
    
#     echo "[GPU 0,1] Done!"
# ) &

# (
#     # echo "[GPU 2,3] Start Distill Qwen 0.5B"
#     # CUDA_VISIBLE_DEVICES=2,3 bash ./scripts/train/qwen2.5-0.5B/distillm_2.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_0.5b.log
    
#     # echo "[GPU 2,3] Start Distill Llama 3.2 3B"
#     # CUDA_VISIBLE_DEVICES=2,3 bash ./scripts/train/llama3.2-3B-it/distillm_2.sh 2>&1 | tee outputs/logs/llama3.2_3b_it_distillm_2.log
    
#     echo "[GPU 2,3] Start Distill Qwen 1.5B"
#     CUDA_VISIBLE_DEVICES=2,3 bash ./scripts/train/qwen2.5-1.5B-it/distillm_2.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_1.5b_it.log
    
#     echo "[GPU 2,3] Done!"
# ) &

(
    # echo "[GPU 4,5] Start Distill-1e Qwen 0.5B"
    # CUDA_VISIBLE_DEVICES=4,5 bash ./scripts/train/qwen2.5-0.5B/distillm_2_1e.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_0.5b-1e.log
    
    # echo "[GPU 4,5] Start Distill-1e Llama 3.2 3B"
    # CUDA_VISIBLE_DEVICES=4,5 bash ./scripts/train/llama3.2-3B-it/distillm_2_1e.sh 2>&1 | tee outputs/logs/llama3.2_3b_it_distillm_2_1e.log
    
    # echo "[GPU 4,5] Start Distill-1e Qwen 1.5B"
    # CUDA_VISIBLE_DEVICES=4,5 bash ./scripts/train/qwen2.5-1.5B-it/distillm_2_1e.sh 2>&1 | tee outputs/logs/distillm_2_qwen2.5_1.5b_1e.log


    echo "========================================="
    echo "Running qwen2.5-0.5B/nnm_distillm_2.sh"
    echo "========================================="
    CUDA_VISIBLE_DEVICES=4,5 bash "./scripts/train/qwen2.5-0.5B/nnm_distillm_2.sh" 2>&1 | tee outputs/logs/nnm_distillm_2_qwen2.5_0.5b_3setup.log


    echo "========================================="
    echo "Running qwen2.5-1.5B-it/nnm_distillm_2.sh"
    echo "========================================="
    CUDA_VISIBLE_DEVICES=4,5 bash "./scripts/train/qwen2.5-1.5B-it/nnm_distillm_2.sh" 2>&1 | tee outputs/logs/nnm_distillm_2_qwen2.5_1.5b_it_3setup.log

    echo "[GPU 4,5] Done!"
) &

(
    echo "========================================="
    echo "Running llama3.2-3B-it/nnm_distillm_2.sh"
    echo "========================================="
    CUDA_VISIBLE_DEVICES=6,7 bash "./scripts/train/llama3.2-3B-it/nnm_distillm_2.sh" 2>&1 | tee outputs/logs/nnm_distillm_2_llama3.2_3b_it_3setup.log
    
    echo "[GPU 6,7] Done!"
) &



wait
echo "Done All!"
