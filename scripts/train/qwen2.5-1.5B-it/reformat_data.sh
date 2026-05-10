
TEACHER_DIR=data/dpo/Qwen/Qwen2.5-14B-Instruct
STUDENT_DIR=data/dpo/Qwen/Qwen2.5-1.5B-Instruct
OUTPUT_DIR=data/reformatted/distill-qwen2.5-14B-Instruct
SFT_OUTPUT_DIR=data/reformatted/sft-qwen2.5-14B-Instruct

python generate/reformat_distill.py --teacher_file $TEACHER_DIR --student_file $STUDENT_DIR --output_dir $OUTPUT_DIR
python generate/reformat_sft.py --teacher_file $TEACHER_DIR --output_dir $SFT_OUTPUT_DIR