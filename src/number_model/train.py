from transformers import LlamaForCausalLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoConfig, DataCollatorWithPadding, GenerationConfig
from torch.utils.data import Dataset, ConcatDataset
from number_model.tokenizer import NumberTokenizer
from number_model.dataset import NumberDataset
from numbers_class import Domain, NumberBasic, DOMAIN_CHOICES
import torch
import time
from transformers.utils import logging
from transformers import EvalPrediction
import numpy as np
from torch.utils.data import DataLoader, IterableDataset, RandomSampler, SequentialSampler
from collections import defaultdict
import pickle
import os
import inspect
from number_model.utils import change_model_config, readable_model_size
import itertools
from compute_metrics import Metrics
from task import Task
import re
from modify_pe import PEModifier
import json
from typing import get_args

logger = logging.get_logger(__name__)

def main(args) -> None:
    # detect whether the code is run on a RTX-4000 GPU
    if re.search(r"40\d0", torch.cuda.get_device_name()) is not None:
        # set two environ `NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1"` to avoid the error
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
    
    if args.multi_task_config is not None:
        with open(args.multi_task_config) as f:
            task_domain_path: list[list[str]] = json.load(f)
    elif args.task is not None and args.domain is not None and args.data_path is not None:
        task_domain_path = [[args.task, args.domain, args.data_path]]
    else:
        raise ValueError("Either a json file containing a [task, domain, data_path] lists or the args.task, args.domain and args.data_path should be given.")
    
    # 1. Create the tokenizer
    tasks = [Task.auto_task(task, domain) for task, domain, _ in task_domain_path] # type: ignore
    multi_task = len(tasks) > 1 
    
    if multi_task:
        raise NotImplementedError("Multi-task training is not supported now.")
    
    tokenizer_param = {
        "task": tasks[0] if not multi_task else None,
        "digit": args.digit,
        "direct": args.direct,
        "direct_float_int": args.direct_float_int,
        "direct_float_dec": args.direct_float_dec,
        "direct_frac_num": args.direct_frac_num,
        "direct_frac_den": args.direct_frac_den,
        "direct_exp": args.direct_exp,
        "direct_mantissa_int": args.direct_mantissa_int,
        "direct_mantissa_dec": args.direct_mantissa_dec,
        "reverse_rep": args.reverse,
        "index_hint": args.index_hint,
        "max_index": args.max_index,
        "hint_digit": args.hint_digit,
        "index_hint_shift_start": True,
        "index_hint_in_answer": False,
        "number_pad": args.pad,
        "random_shifted_start": args.random_shifted_start,
        "random_seed": args.random_seed,
        "random_position": args.random_position
    } 
    
    tokenizer = NumberTokenizer(**tokenizer_param, padding_side="right")
    # random position will never be used in evaluation
    tokenizer_param["random_position"] = False
    # tokenizer_param["index_hint_shift_start"] = False
    eval_tokenizer = NumberTokenizer(**tokenizer_param, padding_side="left")
    
    # 2. Load the dataset
    train_datasets: list[Dataset] = []
    valid_datasets: list[Dataset] = []
    test_datasets: list[Dataset] = []
    for (task, (_, _, data_path)) in zip(tasks, task_domain_path):
        test_d = NumberDataset(
            data=pickle.load(open(os.path.join(data_path, "test.pkl"), "rb")),
            task=task,
            tokenizer=eval_tokenizer,
            training=False,
            remove_duplicate=True,
        )
        valid_d = NumberDataset(
            data=pickle.load(open(os.path.join(data_path, "valid.pkl"), "rb")),
            task=task,
            tokenizer=eval_tokenizer,
            training=False,
            remove_duplicate=True,
            setsub_datasets=[test_d],
        )
        train_d = NumberDataset(
            data=pickle.load(open(os.path.join(data_path, "train.pkl"), "rb")),
            task=task,
            tokenizer=tokenizer,
            training=True,
            remove_duplicate=True,
            setsub_datasets=[test_d, valid_d]
        )
        train_datasets.append(train_d)
        valid_datasets.append(valid_d)
        test_datasets.append(test_d)

    train_dataset: Dataset = ConcatDataset(train_datasets)
    valid_dataset: Dataset = ConcatDataset(valid_datasets)
    test_dataset: Dataset = ConcatDataset(test_datasets)
        

    # 3. define new data collator and then bind them into trainer
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, # type: ignore
        padding=True,
        max_length=args.max_length,
    )

    eval_data_collator = DataCollatorWithPadding(
        tokenizer=eval_tokenizer, # type: ignore
        padding=True,
        max_length=args.max_length,
    )

    # 4. create or load the model
    model_path = args.model_path
    model_config = AutoConfig.from_pretrained(model_path)
    model_config = change_model_config(model_config, tokenizer)
    if args.pe == 'alibi':
        model_config._attn_implementation = "eager"
    if args.checkpoint is None:
        model = LlamaForCausalLM(config=model_config)
    else:
        model = LlamaForCausalLM.from_pretrained(args.checkpoint, config=model_config)
    # model = LlamaForCausalLM.from_pretrained("results/2024-06-27-21-36-02/checkpoint-15000", config=model_config)
    # 4.1 change the pe (if needed)
    pe_modifier = PEModifier(args.pe, non_continuous_pe=args.random_position)
    pe_modifier(model)

    # 5. create the trainer

    import accelerate
    ps = accelerate.PartialState()
    if ps.is_main_process:
        print(f'Number of parameters: {model.num_parameters()}', flush=True)
    curtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    ps.wait_for_everyone()
    curtime = accelerate.utils.broadcast_object_list([curtime], from_process=0)[0]
    ps.wait_for_everyone()

    generation_config = GenerationConfig.from_pretrained(model_path)
    
    metrics = Metrics(tokenizer=eval_tokenizer)
    assert callable(metrics.compute_metrics) and len(list(inspect.signature(metrics.compute_metrics).parameters.keys())) == 1
    compute_metrics = metrics.compute_metrics

    
    model_size = readable_model_size(model.num_parameters())
    # make it readable, as xxk or xxM or xxB
    if args.marks is None:
        args.marks = []

    output_dir = os.path.join(args.output_path, f"task_{args.task}", model_size, f"digit_{args.digit}", f"pe_{args.pe}", f"reverse_{args.reverse}", f"index_hint_{args.index_hint}", f"pad_{args.pad}", *args.marks, curtime)
    log_dir = os.path.join(args.log_path, f"task_{args.task}", model_size, f"digit_{args.digit}", f"pe_{args.pe}", f"reverse_{args.reverse}", f"index_hint_{args.index_hint}", f"pad_{args.pad}", *args.marks, curtime)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, # output directory
        overwrite_output_dir=True,
        num_train_epochs=1, # total number of training epochs
        learning_rate=args.lr, # learning rate
        per_device_train_batch_size=args.batchsize, # batch size per device during training
        per_device_eval_batch_size=args.eval_batchsize, # batch size for evaluation
        warmup_steps=500, # number of warmup steps for learning rate scheduler
        lr_scheduler_type="cosine",
        weight_decay=0.01, # strength of weight decay
        eval_steps=args.eval_steps,
        logging_dir=log_dir, # directory for storing logs,
        save_steps=args.save_steps,
        save_strategy='steps',
        save_total_limit=2,
        data_seed=args.random_seed,
        bf16=True,
        include_inputs_for_metrics=True,
        predict_with_generate=True,
        report_to = 'tensorboard',
        logging_steps=20,
        logging_strategy='steps',
        generation_config=generation_config,
        # resume_from_checkpoint="results/2024-06-27-21-36-02/checkpoint-15000",
        resume_from_checkpoint=args.checkpoint,
    )
    # set eval_strategy or evaluation_strategy as 'steps'
    if hasattr(training_args, "eval_strategy"):
        training_args.eval_strategy = "steps"
    elif hasattr(training_args, "evaluation_strategy"):
        training_args.evaluation_strategy = "steps"
    else:
        raise ValueError("Cannot find `eval_strategy` or `evaluation_strategy` attribute in `training_args`.")

    # 6. Bind the new eval data collator to the eval (and test) dataloader
    # We do this by:
    # (1) re-defining the function get_eval_dataloader & get_test_dataloader
    # (2) binding these two new functions to our Trainer. 
    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = eval_data_collator # NOTE: We change the data collator 

        data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        return self.accelerator.prepare(eval_dataloader)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = eval_data_collator # NOTE: We change the data collator 

        data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    import types
    trainer.get_eval_dataloader = types.MethodType(get_eval_dataloader, trainer)
    trainer.get_test_dataloader = types.MethodType(get_test_dataloader, trainer)
    
    # 6.1 save the args and config into output_dir
    save_dict = vars(args)
    save_dict["trainer_args"] = training_args.to_dict()
    save_dict["model_config"] = model_config.to_dict()
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    with open(os.path.join(os.path.abspath(output_dir), "args.json"), "w") as f:
        json.dump(save_dict, f, indent=4)
        
    # 7. start training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

import argparse
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_task_config", type=str, default=None, help="A file that contain tasks, domains, and corresponding data_paths.")
    parser.add_argument("-d", '--domain', type=str, choices=list(DOMAIN_CHOICES), default=None, help='Which type of number will be learned.')
    parser.add_argument('-p', '--data_path', type=str, default=None, help='The path of data.')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='The path of model config, modify the size of model in the config file.')
    parser.add_argument('-b','--batchsize', type=int, default=64, help='Training batchsize per device')
    parser.add_argument('--eval_batchsize', type=int, default=16, help='Eval batchsize per device')
    parser.add_argument('--output_path', type=str, default='results', help='The path to save the logs.')
    parser.add_argument('--log_path', type=str, default='logs', help="The path to log.")
    parser.add_argument("-t", "--task", type=str, choices=Task.all_tasks, default=None, help="Which task our model will be trained.")
    parser.add_argument("-l", "--lr", type=float, default=5e-5, help='learning rate (considering use smaller lr with larger model)')
    parser.add_argument("--pe", type=str, default="rope", choices=["rope", "nope", "alibi"], help="The type of positional embedding.")
    parser.add_argument("--eval_steps", type=int, default=100, help='eval steps')
    parser.add_argument("--save_steps", type=int, default=500, help='save steps')
    parser.add_argument('-L', '--max_length', type=int, default=None, help="The maximal tokens in a batch. The training samples will truncated. If None, pad into maximal length in a batch.")
    parser.add_argument("--digit", type=int, default=1, help="How many digits consists of a token in our tokenizer.")
    parser.add_argument("--direct", type=str, choices=['right', 'left', 'random'], default='right', help='which direction the tokenizer will tokenize a number. For example, if right, the int 10000 will be tokenized as 10 and 000 when the digit is 3. If left, it will be tokenized as 100 and 00. If random, the string of number will be first randomly split into parts with length smaller than or equal to digit. For example, 10, 00, 0 is a possible split.' )
    parser.add_argument("--direct_float_int", type=str, choices=['right', 'left', 'random', None], default=None, help='Which direction the "Integer" part of a Float will be tokenied from. Default: use `direct`')
    parser.add_argument('--direct_float_dec', type=str, choices=['right', 'left', 'random'], default='left', help='Which direction the "Decimal" part of a Float will be tokenized from.')
    parser.add_argument("--direct_frac_num", type=str, choices=['right', 'left', 'random', None], default=None, help='Which direction the "Numerator" part of a Fraction will be tokenied from. Default: use `direct`')
    parser.add_argument("--direct_frac_den", type=str, choices=['right', 'left', 'random', None], default=None, help='Which direction the "Denominator" part of a Fraction will be tokenied from. Default: use `direct`')
    parser.add_argument('--direct_exp', type=str, choices=['right', 'left', 'random', None], default=None, help='Which direction the "Exponential" part (as an Integer) in a scientific notation will be tokenied from. Default: use `direct`')
    parser.add_argument('--direct_mantissa_int',type=str, choices=['right', 'left', 'random', None], default=None, help='Which direction the "Integer" part of the mantissa in a scientific notation will be tokenied from. Default: use `direct`' )
    parser.add_argument('--direct_mantissa_dec', type=str, choices=['right', 'left', 'random', None], default=None, help='Which direction the "Decimal" part of the mantissa in a scientific notation will be tokenized from. Default: use `direct_float_dec`')
    parser.add_argument('--reverse', type=str, choices=['no', 'each', 'total', 'int', 'dec'], default='no', help='Whether to reverse the string of number. If `each`, reverse the `int` part and `dec` part respectively. For example, 12.19 -> 21.91. If `total`, reverse the whole string. For example, 12.19 -> 91.21. If `int`, reverse the `int` part. For example, 12.19 -> 21.19. If `dec`, reverse the `dec` part. For example, 12.19 -> 12.91.')
    parser.add_argument('-i', '--index-hint', action='store_true', help='Whether to add index hint in the input.')
    parser.add_argument('--max_index', type=int, default=30, help='The maximal digit of EACH PART of a number, only needed when using index hint.')
    parser.add_argument('--hint_digit', type=str, choices=['low', 'high'], default='low', help='The digit of index hint for a token with >1 digit. If low, the hint will be the lowest (most-right) digit. If high, the hint will be the highest (most-left) digit. For example, for number 12345 and digit 2 from right (1,23,45), the hint will be e1c23a45 if low and e1d23b45 if high.')
    parser.add_argument("--pad", action="store_true", help="If True, different numbers will be pad to the same length with 0.")
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help="The path of checkpoint, if None, train from scratch.")
    parser.add_argument("--random_seed", type=int, default=202408130)
    parser.add_argument("--random_shifted_start", action="store_true")
    parser.add_argument("--random_position", action="store_true", help="If True, the position_ids will be a random increasing sequence instead of range(L).")
    parser.add_argument("--marks", nargs='*', type=str, help="Other marks about this running. Will be added as a suffix of saving path.")
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
