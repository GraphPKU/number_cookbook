from numbers_class import Domain, NumberBasic
from collections import defaultdict
from task import Task
from text_model.nl_dataset import NLDataset

import tqdm
import json
import os

from typing import Literal, get_args


class StringMetrics:
    def __init__(self):
        self._set_default_read_direct()
        
    def _set_default_read_direct(self) -> None:
        self.default_read_direct: dict[Domain | Literal["int"], list[Literal['left', 'right']]] = {}
        for domain in get_args(Domain):
            self.default_read_direct[domain] = NumberBasic.get_subclass(domain).default_read_direct()
        self.default_read_direct["int"] = ["right"]
        
    def _clean_str_and_part(self, str_: str, number_type: Domain | Literal['int']) -> list[str]:
        try:
            if number_type == "Integer" or number_type == "int":
                return ["".join([c for c in str_ if c.isdigit()])]
            elif number_type == "Float":
                parts = str_.split(".")
                return ["".join([c for c in parts[0] if c.isdigit()]), "".join([c for c in parts[1] if c.isdigit()])]
            elif number_type == "Fraction":
                parts = str_.split("/")
                return ["".join([c for c in parts[0] if c.isdigit()]), "".join([c for c in parts[1] if c.isdigit()])]
            elif number_type == "ScientificNotation":
                parts = str_.split("e")
                parts = parts[0].split(".") + [parts[1]]
                return ["".join([c for c in parts[0] if c.isdigit()]), "".join([c for c in parts[1] if c.isdigit()]), "".join([c for c in parts[2] if c.isdigit()])]
            else:
                raise ValueError(f"Invalid number type {number_type}")
        except IndexError as e:
            return [""] * 1 if number_type == "Integer" else ([""] * 2 if number_type in ["Float", "Fraction"] else [""] * 3)
        
    def __call__(self, pred_str: str, gt_str: str, expected_type: Domain | Literal['int']):
        pred_parts = self._clean_str_and_part(pred_str, expected_type)
        gt_parts = self._clean_str_and_part(gt_str, expected_type)
        exact_match = int(pred_parts == gt_parts)
        read_direct = self.default_read_direct[expected_type]
        digit_match_c = 0
        for pred_p, gt_p, direct in zip(pred_parts, gt_parts, read_direct):
            if direct == "right":
                pred_p = pred_p[::-1]
                gt_p = gt_p[::-1]
            digit_match_c += sum([int(p == g) for p, g in zip(pred_p, gt_p)])
        digit_match = digit_match_c / sum([len(p) for p in gt_parts])
        dlength = abs(sum(len(p) for p in pred_parts) - sum(len(p) for p in gt_parts))
        return {
            "exact_match": exact_match,
            "digit_match": digit_match,
            "dlength": dlength
        }
        
        
class MetricsRecorder:
    def __init__(self, rfft: bool = False) -> None:
        self.rfft = rfft
        self.value: dict[str, dict[str, defaultdict[int, float]]] = {} # metric_name -> task_name -> digit -> value
        self.count: dict[str, dict[str, defaultdict[int, int]]] = {}
        
    def _record_dict(self, metric_name: str, task_name: str, digit: int, value: float):
        if metric_name not in self.value:
            self.value[metric_name] = {}
            self.count[metric_name] = {}
        if task_name not in self.value[metric_name]:
            self.value[metric_name][task_name] = defaultdict(float)
            self.count[metric_name][task_name] = defaultdict(int)
        self.value[metric_name][task_name][digit] += value
        self.count[metric_name][task_name][digit] += 1
    
    def process(self, processed_file_path: str):
        """
        processed_file_path: str, the path to the processed file, which should be a jsonl file. Each line is dict{"task": str, "digit": int, "generated_text": list[str], "groundtruth": str}
        """
        metrics = StringMetrics()
        with open(processed_file_path, 'r') as rf:
            for line in tqdm.tqdm(rf):
                data = json.loads(line)
                task_name = data["task"]
                _, domain_a, domain_b, domain_output = Task.name2components(task_name)
                digit = data["digit"]
                gt = data["groundtruth"]
                for i, generated_text in enumerate(data["generated_text"]):
                    pred = self.retrieve_answer(generated_text, output_domain = domain_output)
                    metrics_result = metrics(pred, gt, domain_output)
                    for key, value in metrics_result.items():
                        self._record_dict(
                            metric_name=key,
                            task_name=task_name,
                            digit=digit,
                            value=value
                            )
        print('Done!')
        
    def _get_task_digit_gt_for_gpt_batches(self, dataset: NLDataset | None = None, batches_dir: str | None = None) -> dict[int, tuple[str, int, str]]:
        return_dict = {}
        
        if dataset is None:
            assert batches_dir is not None
            for file in os.listdir(batches_dir):
                if not file.endswith(".jsonl"):
                    continue
                with open(os.path.join(batches_dir, file), 'r') as rf:
                    for line in tqdm.tqdm(rf):
                        data = json.loads(line)
                        custom_id = int(data["custom_id"].split("-")[-1])
                        return_dict[custom_id] = (data["metadata"]["task_name"], int(data["metadata"]["digit"]), data["metadata"]["groundtruth"])
        else:
            for i, (task, digit, data, groundtruth) in enumerate(dataset): # type: ignore
                return_dict[i] = (task, digit, groundtruth)
        
        return return_dict                        
                        
        
    def process_gpt_generated(self, generated_file_dir: str, dataset_file: str | None = None):
        import re
        self.gpt_cache: dict[int, str] = {}
        read_tokens = 0
        generate_tokens = 0
        if dataset_file is not None:
            # load the dataset file
            dataset = NLDataset(dataset_file, train_or_test="test", num_each=100, random_seed=20222943)
        else:
            dataset = None
            
        task_digit_gt: dict[int, tuple[str, int, str]] = self._get_task_digit_gt_for_gpt_batches(dataset, "benchmark/gpt_batches")
            
        metrics = StringMetrics()
        
        # process the generated files
        for file in os.listdir(generated_file_dir):
            if not file.endswith(".jsonl"):
                continue
            with open(os.path.join(generated_file_dir, file), 'r') as rf:
                for line in tqdm.tqdm(rf):
                    data = json.loads(line)
                    generated_text = data["response"]["body"]["choices"][0]["message"]["content"]
                    custom_id = int(re.match(r"request-(\d+)", data["custom_id"]).group(1))
                    output_domain = Task.name2components(task_digit_gt[custom_id][0])[3]
                    pred = self.retrieve_answer(generated_text, output_domain=output_domain, start_answer = "The answer is")
                    self.gpt_cache[custom_id] = pred
                    read_tokens += data["response"]["body"]["usage"]["prompt_tokens"]
                    generate_tokens += data["response"]["body"]["usage"]["completion_tokens"]
        
        # process the cache
        for custom_id in range(len(self.gpt_cache)):
            task, digit, gt = task_digit_gt[custom_id]
            pred = self.gpt_cache[custom_id]
            output_domain = Task.name2components(task)[3]
            metrics_result = metrics(pred, gt, output_domain)
            for key, value in metrics_result.items():
                self._record_dict(
                    metric_name=key,
                    task_name=task,
                    digit=digit,
                    value=value
                    )
                
        print("Done! The number of read tokens is", read_tokens, "and the number of generated tokens is", generate_tokens)
        
    def retrieve_answer(self, text: str, output_domain: Domain | Literal["int"], start_answer: str = " = ") -> str:
        if self.rfft:
            try:
                t1 = text.split('So the answer is')
                if len(t1) == 1:
                    t1 = text.split('so the answer is')[1]
                else:
                    t1 = t1[1]
            except:
                t1 = text
            text = t1
        else:
            try:
                text = text[text.index(start_answer) + len(start_answer):]
            except ValueError as e:
                text = text
        text = text.strip()
        if output_domain == "Integer" or output_domain == "int":
            pattern = r"\d+"
        elif output_domain == "Float":
            pattern = r"\d+\.\d+"
        elif output_domain == "Fraction":
            pattern = r"\d+/\d+"
        elif output_domain == "ScientificNotation":
            pattern = r"\d+\.\d+[eE][+-]?\d+"
        else:
            raise ValueError(f"Invalid output domain {output_domain}")
        import re
        match = re.match(pattern, text)
        if match is None:
            return ""
        text = match.group()
        # remove "+" "-" in text and change E to e
        text = text.replace("+", "").replace("-", "").replace("E", "e")
        return text
        
    
    def save(self, save_path: str):
        import pickle
        with open(save_path, 'wb') as wf:
            pickle.dump((self.value, self.count), wf)
        print(f"Save results in {save_path}")
        
    def load(self, load_path: str):
        import pickle
        with open(load_path, 'rb') as rf:
            self.value, self.count = pickle.load(rf)
        print(f"Load results from {load_path}")
        
    def statistics(self, output_file: str | None = None):
        output = {}            
        for metric, task_dict in self.value.items():
            if metric == "exact_match":
                has_performance_thre = 0.1
                well_learned_thre = 0.9
                larger_is_better = True
            elif metric == "digit_match":
                has_performance_thre = 0.5
                well_learned_thre = 0.9
                larger_is_better = True
            else:
                well_learned_thre = 0.1
                has_performance_thre = 1
                larger_is_better = False
                
            record_dict = {}
            for task, digit_dict in task_dict.items():
                well_learned_digit = 0
                has_performance_digit = 0
                
                max_digit = max(digit_dict.keys())
                if max_digit == 21:
                    max_digit = 20 # some bug, to_float has some 21 digit in the dataset
                assert max_digit == 20 or max_digit == 100, f"The maximum digit of a task should be either 20 or 100, but find {max_digit} in task {task}."
                if max_digit == 20:
                    thre = [0, 5, 9, 15 ,21]
                else:
                    thre = [0, 11, 21, 61, 101]
                    
                count_cc = 0 # record for each two range: in-domain and out-domain
                value_cc = 0 # record for each two range: in-domain and out-domain
                averages_range = []
                averages_two_range = []
                for i, (min_digit, max_digit) in enumerate(zip(thre[:-1], thre[1:])):
                    count_c = 0
                    value_c = 0
                    if i == 2:
                        count_cc = 0 # record for each two range
                        value_cc = 0 # record for each two range
                    for digit in range(min_digit, max_digit):
                        if digit in digit_dict:
                            count_c += self.count[metric][task][digit]
                            value_c += self.value[metric][task][digit]
                            count_cc += self.count[metric][task][digit]
                            value_cc += self.value[metric][task][digit]
                            average_digit = self.value[metric][task][digit] / self.count[metric][task][digit]
                            if larger_is_better:
                                if average_digit >= well_learned_thre:
                                    well_learned_digit = max(well_learned_digit, digit)
                                if average_digit >= has_performance_thre:
                                    has_performance_digit = max(has_performance_digit, digit)
                            else:
                                if average_digit <= well_learned_thre:
                                    well_learned_digit = max(well_learned_digit, digit)
                                if average_digit <= has_performance_thre:
                                    has_performance_digit = max(has_performance_digit, digit)
                    if count_c == 0:
                        continue
                    average_range = value_c / count_c
                    averages_range.append(average_range) # length: 4
                    if i == 1 or i == 3:
                        average_two_range = value_cc / count_cc
                        averages_two_range.append(average_two_range) # length: 2
                record_dict[task] = {
                    "well_learned_digit": well_learned_digit,
                    "has_performance_digit": has_performance_digit,
                    "in_domain": averages_two_range[0],
                    "out_domain": averages_two_range[1],
                    "short_range": averages_range[0],
                    "medium_range": averages_range[1],
                    "long_range": averages_range[2],
                    "very_long_range": averages_range[3]
                }
            output[metric] = record_dict
            
            if output_file is not None:
                with open(output_file, 'w') as wf:
                    json.dump(output, wf, indent=2)
        return output
            
    def report(self, digit_range: int | tuple[int, int | None] | None = None, task_name: str | None = None, output_file: str | None = None):
        if digit_range is None:
            every = True
            digit_range = (0, None)
        else:
            every = False
        if isinstance(digit_range, int):
            digit_range = (digit_range, digit_range + 1)
        if digit_range[1] is None:
            digit_range = (digit_range[0], 10000000000)
        assert isinstance(digit_range, tuple) and len(digit_range) == 2, "Invalid digit range"
        if output_file is not None:
            output_file = open(output_file, 'w')
        for metric_name, task_dict in self.value.items():
            print(f"Metric: {metric_name}", file=output_file)
            if task_name is None:
                for task_name_, digit_dict in task_dict.items():
                    print(f"    Task: {task_name_}", file=output_file)
                    value_c = 0
                    count_c = 0
                    
                    if every:
                        for digit, value in digit_dict.items():
                            value_c = value
                            count_c = self.count[metric_name][task_name_][digit]
                            print(f"        Digit: {digit}; Average: {value_c / count_c}", file=output_file)
                    
                    else:
                    
                        for digit, value in digit_dict.items():
                            if digit_range[0] <= digit < digit_range[1]:
                                value_c += value
                                count_c += self.count[metric_name][task_name_][digit]
                        print(f"        Digit Range: [{digit_range[0]}, {digit_range[1]}); Average: {value_c / count_c}", file=output_file)
            else:
                print(f"    Task: {task_name}", file=output_file)
                value_c = 0
                count_c = 0
                if every:
                    for digit, value in task_dict[task_name].items():
                        value_c = value
                        count_c = self.count[metric_name][task_name][digit]
                        print(f"        Digit: {digit}; Average: {value_c / count_c}", file=output_file)
                else:
                    for digit, value in task_dict[task_name].items():
                        if digit_range[0] <= digit < digit_range[1]:
                            value_c += value
                            count_c += self.count[metric_name][task_name][digit]
                    print(f"        Digit Range: [{digit_range[0]}, {digit_range[1]}); Average: {value_c / count_c}", file=output_file)