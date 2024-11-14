from benchmark_test import MetricsRecorder

import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, default="benchmark/test_results")
parser.add_argument("--save_path", type=str, default="benchmark/test_results_report")
parser.add_argument("--regenerate", action="store_true")
parser.add_argument("--model", type=str, default=None)
args = parser.parse_args()

if args.model is None:
    # model_name_list = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"] + [re.match(r"generated_(.*)\.jsonl", name).group(1) for name in os.listdir(args.load_path) if name.endswith(".jsonl")]
    model_name_list = [re.match(r"generated_(.*)\.jsonl", name).group(1) for name in os.listdir(args.load_path) if name.endswith(".jsonl")]
else:
    model_name_list = [args.model]

for model_name in model_name_list:
    if args.regenerate or not os.path.exists(os.path.join(args.save_path, f"{model_name}.txt")) or not os.path.exists(os.path.join(args.save_path, f"{model_name}_statistics.txt")):
        mr = MetricsRecorder()
        if "gpt" in model_name:
            try:
                mr.process_gpt_generated(f"benchmark/gpt_test_results/{model_name}", dataset_file="benchmark/tasks/test.json")
            except FileNotFoundError:
                continue
        else:
            mr.process(os.path.join(args.load_path, f"generated_{model_name}.jsonl"))
    if args.regenerate or not os.path.exists(os.path.join(args.save_path, f"{model_name}.txt")):
        mr.report(output_file=os.path.join(args.save_path, f"{model_name}.txt"))
    if args.regenerate or not os.path.exists(os.path.join(args.save_path, f"{model_name}_statistics.txt")):
        mr.statistics(output_file=os.path.join(args.save_path, f"{model_name}_statistics.txt"))