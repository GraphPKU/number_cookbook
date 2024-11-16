from task import Task
import os
import itertools
import subprocess
from number_model.tokenizer import NumberTokenizer, ReverseType
from utils import ReverseType
from number_model.dataset import NumberDataset
import pickle
import tqdm

from typing import Literal

def record(task_name: str, digit: int, string: str, dict_: dict[str, dict[int, list[str]]]) -> None:
    if task_name not in dict_:
        dict_[task_name] = {}
    if digit not in dict_[task_name]:
        dict_[task_name][digit] = []
    dict_[task_name][digit].append(string)
    
def process(task: Task, dataset_dir: str, name: Literal['train', 'valid', 'test'], subprocess_idx: int, continue_: bool, reverse: ReverseType, pad: bool, cache_path: str, random_seed: int):
    task_name = task.name()
    if continue_ and os.path.exists(os.path.join(cache_path, task_name, name[:-4]+".pkl")):
        with open(os.path.join(cache_path, task_name, name[:-4]+".pkl"), "rb") as f:
            return pickle.load(f)
    tokenizer = NumberTokenizer(task=task, reverse_rep = reverse, random_seed = random_seed + subprocess_idx * 8198, number_pad=pad)
    f = open(os.path.join(dataset_dir, name), "rb")
    dataset = NumberDataset(pickle.load(f), tokenizer=tokenizer, task=task, training=True, trunc=None, return_numbers=True)
    target_dict: dict[str, dict[int, list[str]]] = {}
    import typing
    from number_model.dataset import DataSample
    for i in tqdm.tqdm(range(len(dataset)), desc=f"Solving {str(task)} {name[:-4]}", leave=False):
        d = typing.cast(DataSample, dataset[i]) # because `return_numbers=True`
        tokens = d["tokens"]
        input_number = d["numbers"]
        # input_number = tokenizer.recover_input_number(tokens)
        if isinstance(input_number, tuple):
            digit = max(input_number[0].digit, 0 if isinstance(input_number[1], int) else input_number[1].digit)
        else:
            digit = input_number.digit
        str_ = tokenizer.export(tokens)
        record(
            task_name=task_name, 
            digit=digit, 
            string=str_, 
            dict_=target_dict
            )
    task_name = task.name()
    os.makedirs(os.path.join(cache_path, task_name), exist_ok=True)
    with open(os.path.join(cache_path, task_name, name[:-4]+".pkl"), "wb") as wf:
        pickle.dump((name, subprocess_idx, target_dict), wf)
    return name, subprocess_idx, target_dict

def decon(task):
    """Because the preprocession of the dataset could introduce some duplicate samples, we need to remove them. This function is used to remove test samples in the valid set and remove valid & test samples in the training set."""
    task_name, digit, train, valid, test = task
    if test is not None:
        decon_test = list(set(test))
    else:
        decon_test = None
    if valid is not None and test is not None:
        decon_valid = [txt for txt in valid if txt not in set(decon_test)]
    elif valid is not None:
        decon_valid = list(set(valid))
    else:
        decon_valid = None
    valid_and_test = set(decon_valid if decon_valid is not None else []) | set(decon_test if decon_test is not None else [])
    if train is not None and (valid_and_test):
        decon_train = [txt for txt in train if txt not in valid_and_test]
    elif train is not None:
        decon_train = list(set(train))
    else:
        decon_train = None
    return (task_name, digit, decon_train, decon_valid, decon_test)
    
def main() -> None:
    import argparse
    import json
    from number_model.tokenizer import REVERSE_CHOICES
    parser = argparse.ArgumentParser("""Generate the number and text datasets.""")
    parser.add_argument("-d", "--dataset", action="store_true", help="Whether generate the number dataset. If not, the subfolder `numbers` in the benchmark folder should be prepared.")
    parser.add_argument("-t", "--task", action="store_true", help="Whether generate the text data.")
    parser.add_argument("--decon", action="store_true", help="Whether to post-decontainment of the dataset.")
    parser.add_argument('-r', "--reverse", type=str, default='no', choices=REVERSE_CHOICES, help="Whether reverse the number string representation.")
    parser.add_argument('-p', '--pad', action="store_true")
    parser.add_argument("-c", "--continue_", action="store_true", help="continue the last task")
    args = parser.parse_args()
    
    from datagen.benchmark_config import benchmark_path
    save_path = os.path.join(benchmark_path, "tasks" + ("_pad" if args.pad else "") + ("_reverse" if args.reverse != "no" else ""))
    cache_path = os.path.join(benchmark_path, "cache" + ("_pad" if args.pad else "") + ("_reverse" if args.reverse != "no" else ""))
    if args.dataset:
        from datagen.benchmark_config import dataset_list
        processes = []
        for dataset in dataset_list:
            path = dataset[dataset.index("--save_path")+1]
            if args.continue_:
                if os.path.exists(path):
                    print("Skip", path)
                    continue
            print("Generating dataset", path)
            processes.append(subprocess.Popen("python -m src.datagen.data_generate" + " ".join(dataset), env=os.environ.copy(), shell=True))
        # wait for all the dataset is generated
        for p in processes:
            p.wait()
        print("All dataset is generated.")
    
    if args.task:
        from datagen.benchmark_config import task_dataset_list, random_seed
        import multiprocessing
        with multiprocessing.Pool(processes=3 * len(task_dataset_list)) as pool:
            task_list = list(map(lambda x: (x[1][0][0], x[1][0][1], x[1][1], x[0], args.continue_, args.reverse, args.pad, cache_path, random_seed), enumerate(itertools.product(
                task_dataset_list, ["train.pkl", "valid.pkl", "test.pkl"]
            ))))
            results = pool.starmap(process, task_list)
            
        results.sort(key = lambda x: x[1])
        
        train_dict: dict[str, dict[int, list[str]]] = {}
        valid_dict: dict[str, dict[int, list[str]]] = {}
        test_dict: dict[str, dict[int, list[str]]] = {}
        
        for result in results:
            name, _, target_dict = result
            if name.endswith("train.pkl"):
                for task_name, digit_dict in target_dict.items():
                    if task_name not in train_dict:
                        train_dict[task_name] = {}
                    for digit, string_list in digit_dict.items():
                        if digit not in train_dict[task_name]:
                            train_dict[task_name][digit] = []
                        train_dict[task_name][digit] += string_list
            elif name.endswith("valid.pkl"):
                for task_name, digit_dict in target_dict.items():
                    if task_name not in valid_dict:
                        valid_dict[task_name] = {}
                    for digit, string_list in digit_dict.items():
                        if digit not in valid_dict[task_name]:
                            valid_dict[task_name][digit] = []
                        valid_dict[task_name][digit] += string_list
            elif name.endswith("test.pkl"):
                for task_name, digit_dict in target_dict.items():
                    if task_name not in test_dict:
                        test_dict[task_name] = {}
                    for digit, string_list in digit_dict.items():
                        if digit not in test_dict[task_name]:
                            test_dict[task_name][digit] = []
                        test_dict[task_name][digit] += string_list
        
        if args.decon:
            assert train_dict.keys() == valid_dict.keys() == test_dict.keys(), "The training, valid and test dataset should have the same tasks."
            tasks = []
            for task in train_dict:
                train_digit_dict = train_dict[task]
                valid_digit_dict = valid_dict[task]
                test_digit_dict = test_dict[task]
                digit_list = set(train_digit_dict.keys()) | set(valid_digit_dict.keys()) | set(test_digit_dict.keys())
                tasks += [(task, d, train_digit_dict.get(d, None), valid_digit_dict.get(d, None), test_digit_dict.get(d, None)) for d in digit_list]

            with multiprocessing.Pool(32) as pool:
                l = pool.map(decon, tasks, chunksize=2)
            
            train_dict = {}
            valid_dict = {}
            for returned in l:
                task_name, digit, decon_train, decon_valid, decon_test = returned
                if task_name not in train_dict:
                    train_dict[task_name] = {}
                    valid_dict[task_name] = {}
                    test_dict[task_name] = {}
                if decon_train is not None:
                    train_dict[task_name][digit] = decon_train
                if decon_valid is not None:
                    valid_dict[task_name][digit] = decon_valid
                if decon_test is not None:
                    test_dict[task_name][digit] = decon_test
                    
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "train.json"), "w") as wf:
            json.dump(train_dict, wf, indent=2)
        with open(os.path.join(save_path, "valid.json"), "w") as wf:
            json.dump(valid_dict, wf, indent=2)
        with open(os.path.join(save_path, "test.json"), "w") as wf:
            json.dump(test_dict, wf, indent=2)
    print("Done!")
    
if __name__ == "__main__":
    main()
