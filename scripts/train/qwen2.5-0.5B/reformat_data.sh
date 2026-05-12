
TEACHER_DIR=data/dpo/Qwen/Qwen2.5-Math-1.5B-Instruct
STUDENT_DIR=data/dpo/Qwen/Qwen2.5-0.5B
OUTPUT_DIR=data/reformatted/distill-qwen2.5-Math-1.5B-Instruct
SFT_OUTPUT_DIR=data/reformatted/sft-qwen2.5-Math-1.5B-Instruct

python generate/reformat_distill.py --teacher_file $TEACHER_DIR --student_file $STUDENT_DIR --output_dir $OUTPUT_DIR
python generate/reformat_sft.py --teacher_file $TEACHER_DIR --output_dir $SFT_OUTPUT_DIR