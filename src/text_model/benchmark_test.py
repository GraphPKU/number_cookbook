from torch.utils.data import Dataset, DataLoader
import os
import json
import torch
from task import Task
import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from text_model.nl_dataset import NLDataset
    
def collate_fn(batch: list[tuple[str, int, str, str]]) -> tuple[list[str], list[int], list[str], list[str]]:
    tasks, digits, data, groundtruth = zip(*batch)
    return list(tasks), list(digits), list(data), list(groundtruth)

def gpt_query(data: str, answer_domain: str, idx: int, model: str, groundtruth: str, task_name: str, digit: int) -> dict:
    system_message = \
    """You are a capable math assistant.
    Return your solution without any process in the format: The answer is [YOUR ANSWER].
    The final answer must strictly match the format """
    format_message = {
        "Integer": r'r"\d+"',
        "Float": r'r"\d+\.\d+"',
        "Fraction": r'r"\d+/\d+"',
        "ScientificNotation": r'r"\d+\.\d+e\d+"',
    }
    
    system_message += format_message[answer_domain]
    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": data}]
    return {
        "custom_id": f"request-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "max_tokens": 256,
        },
    }
    
def create_batches_gpt(dataset: NLDataset, model: str, batch_size: int = 40000, batches_path: str = 'benchmark/gpt_batches') -> int:
    batch_idx = 0
    os.makedirs(batches_path, exist_ok=True)
    for i, (task, digit, data, groundtruth) in enumerate(dataset):
        if i % batch_size == 0:
            batch = []
        answer_domain = (Task.name2components(task)[-1]).strip()
        batch.append(gpt_query(data=data, answer_domain=answer_domain, idx=i, model=model, groundtruth=groundtruth, task_name=task, digit=digit))
        if i % batch_size == batch_size - 1:
            with open(os.path.join(batches_path, f"batch_{batch_idx}.jsonl"), 'w') as f:
                for query in batch:
                    f.write(json.dumps(query) + '\n')
            batch_idx += 1
    if len(batch) > 0:
        with open(os.path.join(batches_path, f"batch_{batch_idx}.jsonl"), 'w') as f:
            for query in batch:
                f.write(json.dumps(query) + '\n')
                batch_idx += 1
    return batch_idx
                
def main_gpt_test(model: str, dataset_path: str, num_each: int | None = 100, create_batches: bool = False, create_requests: bool = False, num_batches: int | None = None, random_seed: int = 20222943, report_path: str = "benchmark/gpt_request_report/", batches_path: str = "benchmark/gpt_batches/"):
    # model = "gpt-4o-mini-2024-07-18"
    # model = "gpt-4o-2024-08-06"
    if create_batches:
        dataset = NLDataset(dataset_path, train_or_test="test", num_each=num_each, random_seed=random_seed)
        num_batches = create_batches_gpt(dataset, model=model, batches_path=batches_path)
    if create_requests:
        assert num_batches is not None, "If not creating batches, please provide the number of batches"
        from openai import OpenAI
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("Please set the environment variable OPENAI_API_KEY.")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        os.makedirs(report_path, exist_ok=True)
        with open(os.path.join(report_path, f"record_gpt_requests_{model}.txt"), "w") as wf:
            for i in range(num_batches):
                batch_input_file = client.files.create(
                    file = open(os.path.join(batches_path, f"batch_{i}.jsonl"), 'rb'),
                    purpose="batch"
                )
                batch_input_file_id = batch_input_file.id
                response = client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint = "/v1/chat/completions",
                    completion_window = "24h"
                )
                wf.write(f"Batch idx {i} with id {response.id}\n")
            
def main_test(dataset_path: str, model_name_or_path: str, batchsize: int, num_each: int | None = None, continue_: bool = False, checkpoint: str | None = None, load_in_4_bit: bool = False, reverse: bool = False, pad: bool = False, one_digit_tokenizer: bool = False, nope: bool = False, suffix: str | None = None, rfft: bool = False, save_path: str = "benchmark/test_results", random_seed: int = 20222943):
    if nope:
        from modify_pe import PEModifier
        PEModifier("nope")(None)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if load_in_4_bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config if load_in_4_bit else None)
    
    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, checkpoint)
        # model = model.merge_and_unload()
        print(f"Successfully load checkpoint {checkpoint}")
        
    if reverse:
        dataset_path = dataset_path.replace("tasks", "tasks_reverse")
    if pad:
        dataset_path = dataset_path.replace("tasks", "tasks_pad")
    
    test_dataset = NLDataset(dataset_path, train_or_test="test", num_each=num_each, random_seed=random_seed, rfft=rfft)
    dataloader = DataLoader(test_dataset, batch_size=batchsize, collate_fn=collate_fn, shuffle=False)
    num_return_sequences = 1
    
    if os.path.exists(save_path) and not os.path.isdir(save_path):
        raise ValueError(f"{save_path} should be a directory.")
    os.makedirs(save_path, exist_ok=True)
    
    save_path = os.path.join(save_path, f"generated_{os.path.split(model_name_or_path)[-1]}.jsonl")
    if checkpoint is not None:
        save_path = save_path[:-6] + f"_{os.path.split(checkpoint)[-1]}.jsonl"
        
    if suffix is not None:
        save_path = save_path[:-6] + suffix + '.jsonl'
    
    if continue_ and os.path.exists(save_path):
        with open(save_path, "r") as rf:
            have_generated = len(rf.readlines())
    else:
        have_generated = 0
    if not continue_ and os.path.exists(save_path):
        raise ValueError(f"{save_path} already exists.")
    
    if one_digit_tokenizer:
        one_digit_converter = json.load(open(os.path.join(model_name_or_path, 'one_digit_converter.json'))) # dict[str, list[int]]
        black_token_list = [int(token_id) for token_id in one_digit_converter.keys()]
    else:
        black_token_list = None
    
    for tasks, digits, texts, groundtruths in tqdm.tqdm(dataloader):
        if have_generated >= len(tasks):
            have_generated -= len(tasks)
            continue
        if have_generated > 0:
            tasks = tasks[have_generated:]
            digits = digits[have_generated:]
            texts = texts[have_generated:]
            groundtruths = groundtruths[have_generated:]
            have_generated = 0
        
        if one_digit_tokenizer:
            new_inputs_ids_list: list[list[int]] = [] # the final batch
            for text in texts:
                input_ids: list[int] = tokenizer(text)["input_ids"]
                new_ids = []
                for token_id in input_ids:
                    if str(token_id) not in one_digit_converter:
                        new_ids.append(token_id)
                    else:
                        new_ids.extend(one_digit_converter[str(token_id)])
                        print("debug", token_id, one_digit_converter[str(token_id)])
                new_inputs_ids_list.append(new_ids)
                
            # re-pad them as a batch
            inputs = tokenizer.pad({'input_ids': new_inputs_ids_list}, return_tensors="pt", padding=True) 
        else:
            inputs = tokenizer(texts, return_tensors="pt", padding=True)
        # send to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        # Generate outputs
        with torch.no_grad():
            if rfft:
                max_new_tokens = 2000
            else:
                max_new_tokens = 2 * max(digits)
            bad_words_ids = [black_token_list] * inputs["input_ids"].shape[0] if black_token_list is not None else None
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, bad_words_ids=bad_words_ids) # N_sen * N_return
        
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # regroup the text to align with input
        generated_texts_batch: list[list[str]] = [generated_text[i:i+num_return_sequences] for i in range(0, len(generated_text), num_return_sequences)]
        
        with open(save_path, "a") as wf:
            for task, digit, generated_text_batch, groundtruth in zip(tasks, digits, generated_texts_batch, groundtruths):
                wf.write(json.dumps({
                    "task": task,
                    "digit": digit,
                    "generated_text": generated_text_batch,
                    "groundtruth": groundtruth
                }) + '\n')
                
if __name__ == "__main__":
    int_or_none = lambda x: None if (x == "None" or x == "none") else int(x)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help='dir to save the huggingface model')
    parser.add_argument('--openai', action='store_true', help="whether to test an OpenAI model.")
    parser.add_argument('-p', "--data_path", type=str, default="benchmark/tasks/test.json", help="The path of the test dataset (json file).")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--load_in_4_bit", action="store_true", help="If true, the model is loaded in 4-bit.")
    parser.add_argument("--num", type=int_or_none, default=100, help="number of test samples of each task and length. If None, use all. Default is 100 to save the time.")
    parser.add_argument("--reverse", action="store_true", help="Should only be used when testing a finetuned model with reversed representation.")
    parser.add_argument("--pad", action="store_true", help="Should only be used when testing a finetuned model with zero padding.")
    parser.add_argument("--one_digit_tokenizer", action="store_true", help="Should only be used when test a finetuned model with one digit tokenizer.")
    parser.add_argument("--nope", action="store_true", help="Should only be used when test a finetuned model modified PE as nope.")
    parser.add_argument("--suffix", type=str, default=None, help="A suffix to the generated file name.")
    parser.add_argument("--rfft", action="store_true")
    parser.add_argument("--save_path", type=str, default="benchmark/test_results", help="The directory to save the generated results.")
    args = parser.parse_args()
    
    
    if not args.openai:
        main_test(args.data_path, model_name_or_path=args.model, batchsize=args.batchsize, num_each=args.num, continue_=True, checkpoint=args.checkpoint, load_in_4_bit=args.load_in_4_bit, reverse=args.reverse, pad=args.pad, one_digit_tokenizer=args.one_digit_tokenizer, nope=args.nope, suffix=args.suffix, rfft=args.rfft)
    else:
        main_gpt_test(args.model, args.data_path, num_each=args.num, create_batches=True, create_requests=True)