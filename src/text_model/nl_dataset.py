
from torch.utils.data import Dataset
from typing import Literal
import json
import itertools
import torch
from transformers import PreTrainedTokenizerBase

class NLDataset(Dataset):
    
    def __init__(self, path: str, train_or_test: Literal["train", "test"], num_each: int | None = None, random_seed: int = 20222943, tokenizer: PreTrainedTokenizerBase | None = None, one_digit_converter_file: str | None = None, rfft: bool = False):
        self.path = path
        assert train_or_test in ["train", "test"], "train_or_test must be either 'train' or 'test'"
        self.train = train_or_test == "train"
        self.data_file: dict[str, dict[int, list[str]]] = json.load(open(path, 'r'))
        if num_each is not None:
            self.data_file = self.truncate_data(num_each, random_seed)
        self.num_each = num_each
        self.tasks = list(self.data_file.keys())
        
        if one_digit_converter_file is not None:
            self.one_digit_converter = json.load(open(one_digit_converter_file))
        else:
            self.one_digit_converter = None
        
        # create the global index for each task and the digits in each task
        self.task_digit_indices: list[list[int]] = []
        for task in self.tasks:
            self.task_digit_indices.append([len(self.data_file[task][digit]) for digit in self.data_file[task]])
        self.task_indices = [0] + list(itertools.accumulate([sum(digit_indices) for digit_indices in self.task_digit_indices]))
        self.tokenizer = tokenizer
        self.rfft = rfft
        
    def __len__(self):
        return sum([sum(digit_indices) for digit_indices in self.task_digit_indices])
    
    def truncate_data(self, num_each: int, random_seed: int):
        import random
        random_rng = random.Random(random_seed)
        new_data_file: dict[str, dict[int, list[str]]] = {}
        for task, digits in self.data_file.items():
            new_data_file[task] = {}
            for digit, data in digits.items():
                new_data_file[task][digit] = random_rng.sample(data, min(num_each, len(data)))
        return new_data_file
    
    def __getitem__(self, idx: int) -> tuple[str, int, str, str] | torch.Tensor:
        # find first task idx larger than idx, then this idx - 1 is the task idx
        task_idx = 0
        while idx >= self.task_indices[task_idx + 1]:
            task_idx += 1
        task = self.tasks[task_idx]
        idx -= self.task_indices[task_idx]
        digits = list(self.data_file[task].keys())
        digit_idx = 0
        while idx >= self.task_digit_indices[task_idx][digit_idx]:
            idx -= self.task_digit_indices[task_idx][digit_idx]
            digit_idx += 1
        digit = digits[digit_idx]
        data = self.data_file[task][digit][idx]
        assert "=" in data
        
        # If in train, return the tokenized data
        if self.train:
            assert self.tokenizer is not None, "tokenizer must be provided when training, because we need to tokenize the data"
            if self.one_digit_converter is None:
                return self.tokenizer(data, return_tensors="pt")["input_ids"].squeeze(0)
            else:
                inputs = self.tokenizer(data)
                assert all(i == 1 for i in inputs["attention_mask"])
                new_tokens: list[int] = []
                for index in inputs["input_ids"]:
                    if str(index) not in self.one_digit_converter:
                        new_tokens.append(index)
                    else:
                        new_tokens.extend(self.one_digit_converter[str(index)])
                return torch.tensor(new_tokens)
        
        if self.rfft:
            groundtruth = data.rstrip().split(" ")[-1].strip()
            data = data.split('## Response:')[0] + '## Response:\n'
        else: 
            groundtruth = data.split("=")[1].strip()
            data = data.split("=")[0] + "="
        
        return task, int(digit), data, groundtruth