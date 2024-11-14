import torch
import numpy as np
from number_model.tokenizer import NumberTokenizer
from numbers_class import NumberBasic
from collections import defaultdict
import itertools
from transformers.utils import logging
from transformers import EvalPrediction

from task import Task

logger = logging.get_logger(__name__)


class Metrics:
    def __init__(self, tokenizer: NumberTokenizer):
        self.metrics = ["exact_match", "mse", "dlength", "digit_match", "digit_diff"]
        self.tokenizer = tokenizer
        self.multi_task = tokenizer.multi_task
        self.task = tokenizer.task

    def _init_metrics(self) -> None:
        """It should be called at the beginning of each evaluation."""
        if not self.multi_task:
            assert self.task is not None
            self.record: dict[str, dict[str, dict[int, float]]] = {
                self.task.name(): {metric: defaultdict(float) for metric in self.metrics}
            }
            self.total: dict[str, dict[int, int]] = {self.task.name(): defaultdict(int)}
        else:
            self.record = {}
            self.total = {}

    @staticmethod
    def check_input(input) -> None:
        assert isinstance(input, (torch.Tensor, np.ndarray)) and len(input.shape) == 1, f"Invalid input {input}"

    @staticmethod
    def get_task_digit_label_pred(
        pred: torch.Tensor | np.ndarray,
        input_ids: torch.Tensor | np.ndarray,
        tokenizer: NumberTokenizer,
    ) -> tuple[Task | None, int, NumberBasic | int, NumberBasic | int | None]:
        # calculate the label
        task = tokenizer.recover_task(input_ids)
        inputs_numbers = tokenizer.recover_input_number(
            input_ids
        )  # could be tuple[NumberBasic, NumberBasic] | NumberBasic | tuple[NumberBasic, int]
        if isinstance(inputs_numbers, tuple):
            assert len(inputs_numbers) == 2 and isinstance(
                inputs_numbers[0], NumberBasic
            )
            if isinstance(inputs_numbers[1], NumberBasic):
                # 1.1 if the two-element task
                digit = max(inputs_numbers[0].digit, inputs_numbers[1].digit)
            else:
                # 1.2 otherwise, the second input is a int, which represents an index or something.
                assert isinstance(inputs_numbers[1], int)
                digit = inputs_numbers[0].digit
        else:
            # 1.3 one element operator without index
            assert isinstance(inputs_numbers, NumberBasic)
            digit = inputs_numbers.digit
            inputs_numbers = (inputs_numbers,) # type: ignore
        label: NumberBasic | int = tokenizer.get_answer(list(inputs_numbers), task=task) # type: ignore
        pred_answer: NumberBasic | int | None = tokenizer.retrieve_answer(pred) # `None` means no valid answer has been retrieved
        return task, digit, label, pred_answer

    def compute_each_metric(
        self, metric: str, digit: int, label: NumberBasic | int, pred: NumberBasic | int | None, task_name: str | None
    ) -> None:
        if self.multi_task:
            assert task_name is not None
            if task_name not in self.record:
                assert task_name not in self.total
                self.record[task_name] = {metric: defaultdict(float) for metric in self.metrics}
                self.total[task_name] = defaultdict(int)
        else:
            assert self.task is not None
            task_name = self.task.name()
            assert task_name in self.record and task_name in self.total
        
        assert metric in self.metrics, f"Invalid metric: {metric}"
        assert hasattr(
            self, f"compute_{metric}"
        ), f"Function compute_{metric} not found."
        m: int | float = getattr(self, f"compute_{metric}")(label=label, pred=pred)
        try:
            self.record[task_name][metric][digit] += m
        except OverflowError as e:
            logger.error(f"OverflowError: {e} when calculating {metric} with digit {digit}")
            self.record[task_name][metric][digit] = float('nan')

    def compute_exact_match(self, label: NumberBasic, pred: NumberBasic | None) -> int:
        if pred is not None and pred == label:
            return 1
        return 0

    def compute_mse(self, label: NumberBasic, pred: NumberBasic | None) -> float:
        try:
            if pred is None:
                return label.value**2
            return (pred.value - label.value) ** 2
        except OverflowError as e:
            return float('nan')

    def compute_dlength(self, label: NumberBasic, pred: NumberBasic | None) -> int:
        d_label = label.digit_part
        d_pred_parts = pred.digit_part if pred is not None else [0 for _ in d_label]
        assert len(d_label) == len(d_pred_parts), f"The num of parts of label (type {type(label)}, value: {str(label)}) and pred (type {type(pred)}, value: {str(pred)}) are not equal."
        return sum(abs(d_pred - d_label) for d_label, d_pred in zip(d_label, d_pred_parts))

    def compute_digit_match(
        self, label: NumberBasic, pred: NumberBasic | None
    ) -> float:

        slabel_part = label.part_str()
        spred_part = pred.part_str() if pred is not None else ["" for _ in slabel_part]
        direct = label.default_read_direct()
        for i, d in enumerate(direct):
            if d == "right":
                slabel_part[i] = slabel_part[i][::-1]
                spred_part[i] = spred_part[i][::-1]
        
        return sum(
            sum([int(l == p) for l, p in zip(slabel, spred)])
            for slabel, spred in zip(slabel_part, spred_part)
        ) / sum(len(slabel) for slabel in slabel_part)

    def compute_digit_diff(self, label: NumberBasic, pred: NumberBasic | None) -> float:
        """sum_{i=0^L}(|a_i-b_i|) / L"""
        slabel_part = label.part_str()
        spred_part = pred.part_str() if pred is not None else ["" for _ in slabel_part]
        direct = label.default_read_direct()
        for i, d in enumerate(direct):
            if d == "right":
                slabel_part[i] = slabel_part[i][::-1]
                spred_part[i] = spred_part[i][::-1]
                
        return sum(
            sum(
                [
                    abs(int(l) - int(p))
                    for l, p in itertools.zip_longest(slabel, spred, fillvalue="0")
                ]
            )
            for slabel, spred in zip(slabel_part, spred_part)
        ) / sum(len(slabel) for slabel in slabel_part)

    def _report(self) -> dict[str, float]:
        if not self.multi_task:
            metric_name = lambda task_name, metric, digit: f"{metric}_d{digit}"
        else:
            metric_name = lambda task_name, metric, digit: f"{task_name}_{metric}_d{digit}"
        report: dict[str, float] = {}
        for metric in self.metrics:
            for task_name, task_total in self.total.items():
                for digit, total in task_total.items():
                    try:
                        value = self.record[task_name][metric][digit] / total
                    except OverflowError as e:
                        logger.error(f"OverflowError: {e} when calculating {metric} with digit {digit}")
                        value = float("nan")
                    report[metric_name(task_name, metric, digit)] = value
        return report

    def compute_metrics(self, eval_preds: EvalPrediction):
        self._init_metrics()

        preds = eval_preds.predictions
        inputs_ids = eval_preds.inputs
        # eval_preds has label but it is the language modeling labels (== inputs_ids). For the number label, the label should be calculated from input numbers. 

        for pred, input_ids in zip(preds, inputs_ids): # type: ignore
            if -100 in pred:
                pass
                # logger.warning(f"Found -100 in pred: {pred}")
            pred[pred == -100] = (
                0  # debug, I don't know why there are some -100 in pred
            )

            self.check_input(pred)
            self.check_input(input_ids)

            task, digit, label, pred_answer = self.get_task_digit_label_pred(
                pred, input_ids, self.tokenizer
            )
            
            if task is not None:
                task_name = task.name()
            else:
                assert self.task is not None
                task_name = self.task.name()

            for metric in self.metrics:
                self.compute_each_metric(
                    metric=metric, digit=digit, label=label, pred=pred_answer, task_name = task_name
                )
            self.total[task_name][digit] += 1

        return self._report()
