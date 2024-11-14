from openai import OpenAI
import json

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, required=True)
args = parser.parse_args()

# model_name = "gpt-4o-2024-08-06"
# model_name = "gpt-4o-mini-2024-07-18"
model_name = args.model_name

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set the environment variable OPENAI_API_KEY.")
client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

batch_ids = []

with open(f"benchmark/gpt_request_report/record_gpt_requests_{model_name}.txt") as f:
    for line in f:
        batch_ids.append((line.strip().split(" ")[-1]).strip())

for i, batch_id in enumerate(batch_ids):
    batch = client.batches.retrieve(batch_id)
    if batch.status == "completed":
        output_file_id = batch.output_file_id
        file_response = client.files.content(output_file_id)
        with open(f"benchmark/gpt_test_results/{model_name}/mini_batch_{i}.json", "w") as wf:
            wf.write(file_response.text)
    else:
        print('-'*30)
        print(f"Batch {batch_id} is not completed yet.")
        print(batch)