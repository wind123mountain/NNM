
TEACHER_DIR=data/dpo/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
STUDENT_DIR=data/dpo/meta-llama/Llama-3.2-3B-Instruct
OUTPUT_DIR=data/reformatted/distill-deepSeek-R1-Distill-Llama-8B
SFT_OUTPUT_DIR=data/reformatted/sft-deepSeek-R1-Distill-Llama-8B

python generate/reformat_distill.py --teacher_file $TEACHER_DIR --student_file $STUDENT_DIR --output_dir $OUTPUT_DIR
python generate/reformat_sft.py --teacher_file $TEACHER_DIR --output_dir $SFT_OUTPUT_DIR