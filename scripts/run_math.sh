echo "start"

bash scripts/math/distillm_2_qwen2.5_0.5b.sh > outputs/distillm_2_qwen2.5_0.5b.log 2>&1
bash scripts/math/sft_qwen2.5_0.5b.sh > outputs/sft_qwen2.5_0.5b.log 2>&1
bash scripts/math/nnm_distillm_2_qwen2.5_0.5b.sh > outputs/nnm_distillm_2_qwen2.5_0.5b.log 2>&1

echo "done"