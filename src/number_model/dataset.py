from torch.utils.data import Dataset
from number_model.tokenizer import NumberTokenizer
from numbers_class import NumberBasic
from task import Task

import multiprocessing

import numpy as np

from typing import Sequence, TypedDict, TYPE_CHECKING
import torch
DataSample = TypedDict('DataSample', {'tokens': torch.Tensor, 'numbers': NumberBasic | tuple[NumberBasic, NumberBasic | int]})

class NumberDataset(Dataset):
    def __init__(self, data: list[list[NumberBasic]], tokenizer: NumberTokenizer, task: Task, training: bool = True, trunc: int | None = None, return_numbers: bool = False, remove_duplicate: bool = False, setsub_datasets: Sequence["NumberDataset"] | None = None):
        self.tokenizer = tokenizer
        self.task = task
        self.data = data
        self.training = training
        if trunc is not None:
            self.data = self.data[:trunc]
        self.return_numbers = return_numbers
        
        if remove_duplicate:
            self._remove_duplicate_sample()
        if setsub_datasets is not None:
            setsub_data = sum([dataset.data for dataset in setsub_datasets], start = [])
            self._decontain(reference=setsub_data)
        
    def _remove_duplicate_sample(self):
        with multiprocessing.Pool(32) as p:
            str_pre_tuple = p.map(self.task.preprocess_data, self.data, chunksize=10240)
        str_pre_tuple = [str(pt[0]) + "|" + str(pt[1]) for pt in str_pre_tuple]
        _, index = np.unique(str_pre_tuple, return_index=True)
        self.data = [self.data[i] for i in index]
        
    def _decontain(self, reference: list[list[NumberBasic]]):
        with multiprocessing.Pool(32) as p:
            reference_preprocessed_list = p.map(self.task.preprocess_data, reference, chunksize=10240)
        reference_preprocessed = set(reference_preprocessed_list)
        
        need_remove_index = []
        for i, d in enumerate(self.data):
            a, b = self.task.preprocess_data(d)
            if (a, b) in reference_preprocessed:
                need_remove_index.append(i)
                
        self.data = [self.data[i] for i in range(len(self.data)) if i not in need_remove_index]
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor | DataSample:
        pair = self.data[idx]
        a, b = self.task.preprocess_data(pair)
        input_numbers = (a, b) if b is not None else (a,)
        if not self.return_numbers:
            return self.tokenizer.encode_sample(input_numbers, contain_answer=self.training, return_tensor='pt')
        else:
            return {
                "tokens": self.tokenizer.encode_sample(input_numbers, contain_answer=self.training, return_tensor='pt'),
                "numbers": (a, b) if b is not None else a
            }
    # def __iter__(self):
    #     # give a faster way to iterate the dataset, directly iter the data instead of using __getitem__
    #     for pair in self.data:
    #         a, b = self.task.preprocess_data(pair)
    #         input_numbers = (a, b) if b is not None else (a,)
    #         yield self.tokenizer.encode_sample(input_numbers, contain_answer=self.training, return_tensor='pt')