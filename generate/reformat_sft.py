import os
import json
import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm
import argparse


def main(args):
    dataset = load_dataset('json', data_files=os.path.join(args.teacher_file, 'generated_train.jsonl'), split='train')
    dataset = dataset.rename_columns({
        "prompt": "query",
        "generated_text": "response"
    })
    dataset.save_to_disk(args.output_dir)
    print (f"Binarized datasets save to {os.path.join(args.output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)