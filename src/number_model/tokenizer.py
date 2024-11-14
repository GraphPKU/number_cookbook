import numpy as np
import torch
from typing import Literal, overload, TypeVar, Generic, Type, get_args, ParamSpec, Concatenate, TypedDict
import random
from collections.abc import Sequence, Callable
from numbers import Number
import re

import itertools
import functools

from numbers_class import NumberBasic, Integer, Float, Fraction, ScientificNotation, Domain, MAX_PART_NUM
from datagen.data_utils import to_int, to_list
from task import Task, TaskType

from transformers.utils import logging
from dataclasses import dataclass

from utils import ReverseType, REVERSE_CHOICES

Direct = Literal["right", "left", "random"]

logger = logging.get_logger(__name__)


T = TypeVar('T', bound='NumberTokenizer')
R = TypeVar('R', bound='NumberBasic|int')
P = ParamSpec('P')

class DecodeError(Exception):
    pass

@dataclass
class NumberTokenizer:
    task: Task | None = None
    digit: int = 1
    padding_side: Literal["right", "left"] = "right"
    random_shifted_start: bool = False
    random_shifted_prob: float = 0.7
    random_position: bool = False
    random_min_index: int = 0
    random_max_index: int = 1024
    random_position_prob: float = 1.0
    random_seed: int | None = None
    reverse_rep: ReverseType = 'no'
    index_hint: bool = False
    max_index: int | None = None # for index hint, we need additional hint tokens to denote the index
    hint_digit: Literal['low', 'high'] = 'low'
    index_hint_shift_start: bool = False # Should be True when training. True of False when evaluating.
    index_hint_in_answer: bool | None = None
    direct: Direct = "right"
    direct_float_int: Direct | None = None
    direct_float_dec: Direct = "left"
    direct_frac_num: Direct | None = None 
    direct_frac_den: Direct | None = None 
    direct_mantissa_int: Direct | None = None
    direct_mantissa_dec: Direct | None = None
    direct_exp: Direct | None = None
    number_pad: bool = False
    export_index_hint_vocab: Sequence[str] | None = None
    multi_task: bool = False

    def __post_init__(self):
        if not self.multi_task:
            assert self.task is not None
        else:
            self.task = None
        self.set_rng(random_seed=self.random_seed)
        self.set_addition_tokens()
        if not self.index_hint: 
            self.max_index = None
        else: 
            assert self.max_index is not None, "The `max_index` is needed if index_hint."
        if self.reverse_rep not in REVERSE_CHOICES:
            raise ValueError(f'Invalid `reverse_rep` value {self.reverse_rep}; not in candidate {REVERSE_CHOICES}.')
        self.set_domain_and_task_tokens()

    def set_rng(self, random_seed: int | None = None) -> None:
        self.random_rng = random.Random(random_seed)
        self.np_rng = np.random.default_rng(random_seed)
        self.torch_rng = torch.Generator()
        if random_seed is not None:
            self.torch_rng.manual_seed(random_seed)

    def set_addition_tokens(self) -> None:
        self.addition_token: dict[str, int] = {
            "pad_token": 0,
            "bot_token": 1,
            "eot_token": 2,
            "split_token": 3,  # " + "
            "answer_token": 4,  # " = "
            "part_token": 5,  # " . " in float, "/" in fraction, ...
            # Maybe we need to denote the pos or neg, but now we don't.
            # "pos_token": 6, # positive, if needed
            # "neg_token": 6, # negative sign
        }
        self.addition_token_str: dict[int, str] = {
            0: "<PAD>",
            1: "<BOT>",
            2: "<EOT>",
            3: " <OPE> ",
            4: " = ",
            5: "<s>",
        }  # used for visualization    
        
    def set_domain_and_task_tokens(self) -> None:
        start = self.num_addition_token + self.num_number_token + self.num_hint_token
        self.domain_token: dict[Domain | Literal["int", "none"], int] = {
            key: start + i for i, key in enumerate(get_args(Domain)+('int', 'none'))
        }
        self.task_token: dict[TaskType, int] = {
            key: start + i + len(self.domain_token) for i, key in enumerate(get_args(TaskType))
        }
        self.domain_token_str: dict[int, Domain | Literal["int", "none"]] = {v: k for k, v in self.domain_token.items()}
        self.task_token_str: dict[int, TaskType] = {v: k for k, v in self.task_token.items()}
    
    def get_hint_token(self, part_index: int, digit_index: int) -> int:
        assert part_index < MAX_PART_NUM
        assert self.max_index is not None
        assert digit_index < self.max_index
        hint_token = self.num_addition_token + self.num_number_token + part_index * self.max_index + digit_index
        assert self.is_hint_token(hint_token), f"{hint_token} is not a hint token"
        return hint_token

    @property
    def num_number_token(self) -> int:
        return sum(10 ** (d + 1) for d in range(self.digit))
    
    @property
    def num_hint_token(self) -> int:
        if not self.index_hint:
            return 0
        else:
            assert self.max_index is not None
            return MAX_PART_NUM * self.max_index

    @property
    def num_addition_token(self) -> int:
        return len(self.addition_token)
    
    @property
    def num_task_token(self) -> int:
        if self.multi_task:
            return len(self.task_token)
        return 0
    
    @property
    def num_domain_token(self) -> int:
        if self.multi_task:
            return len(self.domain_token)
        return 0

    @property
    def num_token(self) -> int:
        """tokens in tokenizer: addition_token, number_tokens, hint_tokens, domain_tokens, task_tokens. The latter two could be empty."""
        return self.num_addition_token + self.num_number_token + self.num_hint_token + self.num_domain_token + self.num_task_token
    
    def is_token(self, token: int) -> bool:
        return token >= 0 and token < self.num_token
    
    def is_addition_token(self, token: int) -> bool:
        return self.is_token(token) and (0 <= token < self.num_addition_token)

    def is_number_token(self, token: int) -> bool:
        return (self.is_token(token) and 
                (self.num_number_token + self.num_addition_token) > token >= self.num_addition_token
                )
    
    def is_hint_token(self, token: int) -> bool:
        return self.is_token(token) and (token >= self.num_addition_token + self.num_number_token) and (token < self.num_addition_token + self.num_number_token + self.num_hint_token)
    
    def is_domain_token(self, token: int) -> bool:
        return self.is_token(token) and (token >= self.num_addition_token + self.num_number_token + self.num_hint_token) and (token < self.num_addition_token + self.num_number_token + self.num_hint_token + self.num_domain_token)
    
    def is_task_token(self, token: int) -> bool:
        return self.is_token(token) and (token >= self.num_addition_token + self.num_number_token + self.num_hint_token + self.num_domain_token) and (token < self.num_addition_token + self.num_number_token + self.num_hint_token + self.num_domain_token + self.num_task_token)

    def get_answer(self, sample: Sequence[NumberBasic | int], task: Task | None = None) -> NumberBasic | int:
        if task is None:
            task = self.task
        assert task is not None
        number_a = sample[0]
        assert isinstance(number_a, NumberBasic), "The first input number in a sample should be a NumberBasic."
        if len(sample) == 1:
            return task.calc_result(number_a, None)
        elif len(sample) == 2:
            return task.calc_result(number_a, sample[1])
        else:
            raise ValueError("The sample should contain 1 or 2 numbers.")

    def encode_number(self, input_: NumberBasic | int, index_hint: bool | None = None, index_hint_shifted_start: Sequence[int] | None = None, pad_target_lens: Sequence[int] | None = None) -> list[int]:
        if isinstance(input_, int):
            tokens = self.encode_number_int(input_, index_hint=index_hint, index_hint_shifted_start=index_hint_shifted_start, pad_target_lens=pad_target_lens)
        elif isinstance(input_, NumberBasic):
            tokens =  getattr(self, f"encode_number_{type(input_).__name__.lower()}")(input_, index_hint = index_hint, index_hint_shifted_start=index_hint_shifted_start, pad_target_lens=pad_target_lens)
        else:
            raise TypeError(f"Invalid input type {type(input_)}.")
        return tokens

    def decode_number(
        self, tokens: Sequence[int] | torch.Tensor | np.ndarray, expected_type: Domain | Literal["int"]
    ) -> NumberBasic | int:
        assert expected_type in get_args(Domain) or expected_type == "int", f"Invalid expected_type {expected_type}."
        assert hasattr(self, f"decode_number_{expected_type.lower()}"), f"Method decode_number_{expected_type.lower()} not found."
        return getattr(self, f"decode_number_{expected_type.lower()}")(tokens)
    
    @staticmethod
    def decode_preprocess(func: Callable[Concatenate[T, Sequence[int], P], R]) -> Callable[Concatenate[T, Sequence[int] | torch.Tensor | np.ndarray, P], R]:
        @functools.wraps(func)
        def wrap_func(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
            # find param: tokens
            if "tokens" in kwargs:
                tokens = kwargs["tokens"]
            else:
                tokens = args[0]
                args = args[1:] # type: ignore
            assert isinstance(tokens, (int, Sequence, torch.Tensor, np.ndarray))
            tokens_list = to_list(tokens)
            tokens_list = self.remove_index_hint(tokens_list)
            return func(self, tokens_list, *args, **kwargs)
        return wrap_func # type: ignore

    def encode_number_int(self, int_: int, direct: Direct | None = None, reverse: bool | None = None, index_hint: bool | None = None, index_hint_shifted_start: Sequence[int] | None = None, pad_target_lens: Sequence[int] | int | None = None, allow_negative: bool = False) -> list[int]:
        if not isinstance(int_, int) and int_ < 0 and not allow_negative:
            raise ValueError(
                f"Only non-negative integers are supported, but get a {type(int_)}: {int_}."
            )
        
        direct: Direct = self.direct if direct is None else direct # type: ignore
        reverse: bool = (self.reverse_rep in ["int", "each", "total"]) if reverse is None else reverse
        s_int_ = str(int_)
        if pad_target_lens is not None:
            if isinstance(pad_target_lens, Sequence):
                assert len(pad_target_lens) == 1
                pad_target_len = pad_target_lens[0]
            elif isinstance(pad_target_lens, int):
                pad_target_len = pad_target_lens
            else:
                raise TypeError(f"pad_target_lens should be a Sequence or int, but find {type(pad_target_lens)}")
            s_int_ = ('0' * (pad_target_len - len(s_int_))) + s_int_
        if reverse:
            s_int_ = s_int_[::-1]
        assert direct is not None
        assert isinstance(int_, int)
        if direct == "left":
            tokens = [
                self.encode_one_token(s_int_[i : i + self.digit])
                for i in range(0, len(s_int_), self.digit)
            ]
        elif direct == "right":
            s_int_ = s_int_[::-1]
            tokens = [
                self.encode_one_token(s_int_[i : i + self.digit][::-1])
                for i in range(0, len(s_int_), self.digit)
            ][::-1]
        elif direct == "random":
            tokens_str = []
            while s_int_:
                l = self.random_rng.randint(1, self.digit)
                tokens_str.append(s_int_[:min(l, len(s_int_))])
                s_int_ = s_int_[min(l, len(s_int_)):]
            tokens = [self.encode_one_token(s) for s in tokens_str]
        else:
            raise ValueError(
                f"Invalid direct {direct}. Should be 'right' or 'left'."
            )
        index_hint: bool = index_hint if index_hint is not None else self.index_hint
        if index_hint:
            count_side: Literal['left', 'right'] = "right" if not reverse else "left"
            hint_index: Literal['left', 'right'] = "right" if (reverse and self.hint_digit == "high") or (not reverse and self.hint_digit == "low") else "left"
            tokens = self.add_index_hint(tokens, [count_side], [hint_index], index_hint_shifted_start)
            
        if int_ < 0:
            tokens = [self.addition_token["neg_token"]] + tokens
        return tokens
    
    @decode_preprocess
    def decode_number_int(self, tokens: Sequence[int], reverse: bool | None = None) -> int:
        """For some tasks, it will return int directly."""
        reverse = (self.reverse_rep in ["int", "each", "total"]) if reverse is None else reverse
        tokens = to_list(tokens)
        tokens = self.remove_index_hint(tokens)
        if reverse:
            # first we remove the token list: For example, a given integer 123450 has reverse representation 054321, which is tokenized as ['05', '43', '21']
            # Here we first reverse the list to ['21', '43', '05']
            tokens = tokens[::-1]
        if len(tokens) == 0:
            raise DecodeError("No valid tokens.")
        num: int = 0
        for t in tokens:
            s_int_ = self.decode_one_token(t)
            if reverse:
                # then we reverse each token to ['12', '34', '50']
                s_int_ = s_int_[::-1]
            num *= 10 ** len(s_int_)
            num += int(s_int_)
        assert isinstance(num, int)
        return num
    
    def encode_number_integer(self, input_: Integer, reverse: bool | None = None, index_hint: bool | None = None, index_hint_shifted_start: Sequence[int] | None = None, pad_target_lens: Sequence[int] | None = None) -> list[int]:
        reverse = (self.reverse_rep in ["int", "each", "total"]) if reverse is None else reverse
        enc = self.encode_number_int(input_.value, reverse=reverse, index_hint=index_hint, index_hint_shifted_start=index_hint_shifted_start, pad_target_lens=pad_target_lens)
        return enc
    
    @decode_preprocess
    def decode_number_integer(self, tokens: Sequence[int], reverse: bool | None = None) -> Integer:
        reverse = (self.reverse_rep in ["int", "each", "total"]) if reverse is None else reverse
        num = self.decode_number_int(tokens, reverse=reverse)
        return Integer(num)
    
    def _encode_dec_part(self, dec_part: Sequence[int], direct: Direct | None = None, reverse: bool | None = None, pad_target_len: None | int = None) -> list[int]:
        reverse = (self.reverse_rep in ["dec", "each", "total"]) if reverse is None else reverse
        if pad_target_len is not None:
            dec_part = list(dec_part) + [0] * (pad_target_len - len(dec_part)) 
        if reverse:
            dec_part = dec_part[::-1]
        if direct is None:
            direct = self.direct_float_dec
        if direct == "left":
            str_tokens = [
                "".join(map(str, dec_part[i : i + self.digit]))
                for i in range(0, len(dec_part), self.digit)
            ]
        elif direct == "right":
            dec_part = dec_part[::-1]
            str_tokens = [
                "".join(map(str, dec_part[i : i + self.digit][::-1]))
                for i in range(0, len(dec_part), self.digit)
            ][::-1]
        elif direct == "random":
            str_tokens = []
            while dec_part:
                l = self.random_rng.randint(1, self.digit)
                str_tokens.append("".join(map(str, dec_part[:min(l, len(dec_part))])))
                dec_part = dec_part[min(l, len(dec_part)):]
        else:
            raise ValueError(
                f"Invalid direct_dec {direct}. Should be 'right' or 'left'."
            )
        tokens = [self.encode_one_token(s) for s in str_tokens]
        return tokens

    def encode_number_float(self, input_: Float, direct_int: Direct | None = None, direct_dec: Direct | None = None, reverse_int: bool | None = None, reverse_dec: bool | None = None, reverse_total: bool | None = None, index_hint: bool | None = None, index_hint_shifted_start: Sequence[int] | None = None, pad_target_lens: Sequence[int] | None = None) -> list[int]:
        assert pad_target_lens is None or len(pad_target_lens) == 2
        reverse_int = (self.reverse_rep in ["int", "each", "total"]) if reverse_int is None else reverse_int
        reverse_dec = (self.reverse_rep in ["dec", "each", "total"]) if reverse_dec is None else reverse_dec
        int_part: list[int] = self.encode_number_int(input_.int_part, direct = direct_int if direct_int is not None else self.direct_float_int, reverse=reverse_int, index_hint=False, pad_target_lens=([pad_target_lens[0]] if pad_target_lens is not None else None)) # add index hint before return
        dec_part: list[int] = self._encode_dec_part(input_.dec_part, direct = direct_dec if direct_dec is not None else self.direct_float_dec, reverse=reverse_dec, pad_target_len=(pad_target_lens[1] if pad_target_lens is not None else None))
        reverse_total = (self.reverse_rep == "total") if reverse_total is None else reverse_total
        if reverse_total:
            tokens = dec_part + [self.addition_token["part_token"]] + int_part
        else:
            tokens = int_part + [self.addition_token["part_token"]] + dec_part
        index_hint = index_hint if index_hint is not None else self.index_hint
        if index_hint:
            count_side_int: Literal['left', 'right'] = "right" if not reverse_int else "left"
            count_side_dec: Literal['left', 'right'] = "right" if not reverse_dec else "left"
            count_side = (count_side_int, count_side_dec) if not reverse_total else (count_side_dec, count_side_int)
            hint_index_int: Literal['left', 'right'] = "right" if (reverse_int and self.hint_digit == "high") or (not reverse_int and self.hint_digit == "low") else "left"
            hint_index_dec: Literal['left', 'right'] = "right" if (reverse_dec and self.hint_digit == "high") or (not reverse_dec and self.hint_digit == "low") else "left"
            hint_index = (hint_index_int, hint_index_dec) if not reverse_total else (hint_index_dec, hint_index_int)
            if reverse_total and index_hint_shifted_start is not None:
                index_hint_shifted_start = index_hint_shifted_start[::-1]
            tokens = self.add_index_hint(tokens, count_side, hint_index, index_hint_shifted_start)
        return tokens

    @decode_preprocess
    def decode_number_float(self, tokens: Sequence[int], reverse_int: bool | None = None, reverse_dec: bool | None = None, reverse_total: bool | None = None) -> Float:
        assert len(tokens) > 0
        try:
            split_index = tokens.index(self.addition_token["part_token"])
        except ValueError as e:
            raise DecodeError(f"No part_token has been found in the tokens {tokens} when decoding them as a float.")
        reverse_int = (self.reverse_rep in ["int", "each", "total"]) if reverse_int is None else reverse_int
        reverse_dec = (self.reverse_rep in ["dec", "each", "total"]) if reverse_dec is None else reverse_dec
        reverse_total = (self.reverse_rep == "total") if reverse_total is None else reverse_total
        
        int_tokens = tokens[:split_index]
        dec_tokens = tokens[split_index + 1 :]
        if reverse_total:
            int_tokens, dec_tokens = dec_tokens, int_tokens
        int_part: int = self.decode_number_int(int_tokens, reverse=reverse_int)
        dec_part: list[int] = []
        for t in dec_tokens:
            t_str = self.decode_one_token(t)
            if len(t_str) == 0:
                raise DecodeError(f'Invalid token {t_str}')
            dec_part += map(int, list(t_str))
            if reverse_dec:
                dec_part = dec_part[::-1]
        return Float(int_part, dec_part)
    
    def encode_number_fraction(self, input_: Fraction, index_hint: bool | None = None,  index_hint_shifted_start: Sequence[int] | None = None, pad_target_lens: Sequence[int] | None = None) -> list[int]:
        assert pad_target_lens is None or len(pad_target_lens) == 2
        pad_num = None if pad_target_lens is None else [pad_target_lens[0]]
        pad_den = None if pad_target_lens is None else [pad_target_lens[1]]
        reverse_num = reverse_den = (self.reverse_rep in ["int", "each", "total"])
        num_tokens = self.encode_number_int(input_.num, direct=self.direct_frac_num, reverse = reverse_num, index_hint=False, pad_target_lens=pad_num)
        den_tokens = self.encode_number_int(input_.den, direct=self.direct_frac_den, reverse = reverse_den, index_hint=False, pad_target_lens=pad_den)
        reverse_total = (self.reverse_rep == "total")
        if reverse_total:
            tokens = den_tokens + [self.addition_token["part_token"]] + num_tokens
        else:
            tokens =  num_tokens + [self.addition_token["part_token"]] + den_tokens
        index_hint = index_hint if index_hint is not None else self.index_hint
        if index_hint:
            count_side_num: Literal['left', 'right'] = "right" if not reverse_num else "left"
            count_side_den: Literal['left', 'right'] = "right" if not reverse_den else "left"
            count_side = (count_side_num, count_side_den) if not reverse_total else (count_side_den, count_side_num)
            hint_index_num: Literal['left', 'right'] = "right" if (reverse_num and self.hint_digit == "high") or (not reverse_num and self.hint_digit == "low") else "left"
            hint_index_den: Literal['left', 'right'] = "right" if (reverse_den and self.hint_digit == "high") or (not reverse_den and self.hint_digit == "low") else "left"
            hint_index = (hint_index_num, hint_index_den) if not reverse_total else (hint_index_den, hint_index_num)
            if reverse_total and index_hint_shifted_start is not None:
                index_hint_shifted_start = index_hint_shifted_start[::-1]
            tokens = self.add_index_hint(tokens, count_side, hint_index, index_hint_shifted_start)
        return tokens

    @decode_preprocess
    def decode_number_fraction(self, tokens: Sequence[int]) -> Fraction:
        try:
            split_index = tokens.index(self.addition_token["part_token"])
        except ValueError as e:
            raise DecodeError(f"No part_token has been found in the tokens {tokens} when decoding them as a fraction.")
        num_tokens = tokens[:split_index]
        den_tokens = tokens[split_index + 1 :]
        if self.reverse_rep == "total":
            num_tokens, den_tokens = den_tokens, num_tokens
            
        num_part = self.decode_number_int(
            num_tokens, reverse = (self.reverse_rep in ["int", "each", "total"])
        )
        den_part = self.decode_number_int(
            den_tokens, reverse = (self.reverse_rep in ["int", "each", "total"])
        )
        if den_part == 0:
            raise DecodeError("The denominator of a fraction should not be zero.")
        return Fraction(num_part, den_part)
    
    def encode_number_scientificnotation(self, input_: ScientificNotation, index_hint: bool | None = None, index_hint_shifted_start: Sequence[int] | None = None, pad_target_lens: Sequence[int] | None = None) -> list[int]:
        reverse_man_int = self.reverse_rep in ["int", "each", "total"]
        reverse_man_dec = self.reverse_rep in ["dec", "each", "total"]
        reverse_total = self.reverse_rep == "total"
        reverse_exp = self.reverse_rep in ["int", "each", "total"]
        
        if pad_target_lens is None:
            pad_man = pad_exp = None
        else:
            assert len(pad_target_lens) == 3
            pad_man = pad_target_lens[:2]
            pad_exp = [pad_target_lens[2]]
        
        mantissa_part = self.encode_number_float(input_.mantissa, direct_int=self.direct_mantissa_int, direct_dec=self.direct_mantissa_dec, reverse_int = reverse_man_int, reverse_dec = reverse_man_dec, reverse_total = reverse_total, index_hint = False, pad_target_lens=pad_man)
        exp_part = self.encode_number_int(input_.exponent, direct=self.direct_exp, reverse=reverse_exp, index_hint=False, pad_target_lens=pad_exp)
        if self.reverse_rep == "total":
            tokens = exp_part + [self.addition_token["part_token"]] + mantissa_part
        else:
            tokens = mantissa_part + [self.addition_token["part_token"]] + exp_part
        index_hint = index_hint if index_hint is not None else self.index_hint
        if index_hint:
            count_side_man_int: Literal['left', 'right'] = "right" if not reverse_man_int else "left"
            count_side_man_dec: Literal['left', 'right'] = "left" if not reverse_man_dec else "right"
            count_side_exp: Literal['left', 'right'] = "right" if not reverse_exp else "left"
            count_side = (count_side_man_int, count_side_man_dec, count_side_exp) if not reverse_total else (count_side_exp, count_side_man_dec, count_side_man_int)
            hint_index_man_int: Literal['left', 'right'] = "right" if (reverse_man_int and self.hint_digit == "high") or (not reverse_man_int and self.hint_digit == 'low') else "left"
            hint_index_man_dec: Literal['left', 'right'] = "right" if (reverse_man_dec and self.hint_digit == "high") or (not reverse_man_dec and self.hint_digit == 'low') else "left"
            hint_index_exp: Literal['left', 'right'] = "right" if (reverse_exp and self.hint_digit == "high") or (not reverse_exp and self.hint_digit == 'low') else "left"
            hint_index = (hint_index_man_int,  hint_index_man_dec, hint_index_exp) if not reverse_total else (hint_index_exp,  hint_index_man_dec, hint_index_man_int)
            if reverse_total and index_hint_shifted_start is not None:
                index_hint_shifted_start = index_hint_shifted_start[::-1]
            tokens = self.add_index_hint(tokens, count_side=count_side, hint_index=hint_index, index_hint_shifted_start=index_hint_shifted_start)
        return tokens
            
    @decode_preprocess
    def decode_number_scientificnotation(self, tokens: Sequence[int]) -> ScientificNotation:
        try:
            split1 = tokens.index(self.addition_token["part_token"])
            split2 = tokens.index(self.addition_token["part_token"], split1 + 1)
        except ValueError as e:
            raise DecodeError(f"Less than 2 part_tokens habe been found in the tokens {tokens} when decoding them as a scientific notation.")
        if self.reverse_rep == "total":
            mantissa_tokens = tokens[split1 + 1 :]
            exp_tokens = tokens[: split1]
        else:
            mantissa_tokens = tokens[:split2]
            exp_tokens = tokens[split2 + 1 :]
            
        mantissa_part = self.decode_number_float(mantissa_tokens, reverse_int=(self.reverse_rep in ["int", "each", "total"]), reverse_dec=(self.reverse_rep in ["dec", "each", "total"]), reverse_total=(self.reverse_rep == "total"))
        exp_part = self.decode_number_int(exp_tokens, reverse=(self.reverse_rep in ["int", "each", "total"]))
        try:
            return_ = ScientificNotation(mantissa_part, exp_part)
        except ValueError as e:
            raise DecodeError(e)
        return return_

    PreprocessReturnDict = TypedDict("PreprocessReturnDict", {'part_nums': list[int], 'shifted_start': None | list[int], "pad_target_lens": list[list[int]] | None})
    def preprocess_sample(self, sample: Sequence[NumberBasic | int]) -> PreprocessReturnDict:
        part_nums: list[int] = [s.num_part if isinstance(s, NumberBasic) else 1 for s in sample]
        parts_str: list[list[str]] = [s.part_str() if isinstance(s, NumberBasic) else [str(s)] for s in sample]
        # 1. index_hint, index_hint need to sample a random but continuous int for each part. And we need the max index hint is smaller than the self.max_index.
        if self.index_hint and self.index_hint_shift_start:
            assert self.max_index is not None
            # first, we calculate the random start: keep (start + max(len(part))) < self.max_index for each part and each number
            shifted_start: list[int] = []
            for part_index in range(max(part_nums)):
                max_length_of_this_part = max([len(s[part_index]) if len(s) > part_index else 0 for s in parts_str])
                if max_length_of_this_part > self.max_index:
                    raise ValueError(f"The max length of part {part_index} is {max_length_of_this_part}, which is larger than the max index {self.max_index}.")
                if max_length_of_this_part > 0.7 * self.max_index:
                    logger.warning(f"The max length of part {part_index} is {max_length_of_this_part}, which is larger than 0.7 * max index {self.max_index}.")
                shifted_start.append(self.random_rng.randint(0, self.max_index - max_length_of_this_part))
        else:
            shifted_start = None
            
        # 2. pad_target_lens
        pad_target_lens: list[list[int]] | None 
        if self.number_pad:
            pad_target_lens = [[] for _ in range(len(sample))]
            for part_index in range(max(part_nums)):
                max_length_of_this_part = max([len(s[part_index]) if len(s) > part_index else 0 for s in parts_str])
                for i in range(len(sample)):
                    if part_index < part_nums[i]:
                        pad_target_lens[i].append(max_length_of_this_part)
                        
            assert all([len(ptl) == pn for ptl, pn in zip(pad_target_lens, part_nums)])
        else:
            pad_target_lens = None
        return {
            "part_nums": part_nums,
            "shifted_start": shifted_start,
            "pad_target_lens": pad_target_lens,
        }
        
    def add_domain_and_task_prefix(self, tokens: list[int], task: Task | None) -> list[int]:
        if not self.multi_task:
            return tokens
        else:
            assert task is not None, "If multi_task, the task of each sample should be given."
            domain_a = task.input_a_type
            domain_b = task.input_b_type
            domain_output = task.output_type
            task_name = task.task
            return [self.domain_token[domain_a], self.domain_token[domain_b], self.domain_token[domain_output], self.task_token[task_name]] + tokens
    
    @overload
    def encode_sample(
        self,
        sample: Sequence[NumberBasic | int],
        contain_answer: bool = True,
        task: Task | None = None,
        *,
        return_tensor: Literal["no"] = "no",
    ) -> list[int]: ...
    @overload
    def encode_sample(
        self,
        sample: Sequence[NumberBasic | int],
        contain_answer: bool = True,
        task: Task | None = None,
        *,
        return_tensor: Literal["pt"],
    ) -> torch.Tensor: ...
    @overload
    def encode_sample(
        self,
        sample: Sequence[NumberBasic | int],
        contain_answer: bool = True,
        task: Task | None = None,
        *,
        return_tensor: Literal["np"],
    ) -> np.ndarray: ...
    def encode_sample(
        self,
        sample: Sequence[NumberBasic | int],
        contain_answer: bool = True,
        task: Task | None = None,
        *,
        return_tensor: Literal["no", "pt", "np"] = "no",
    ) -> list[int] | np.ndarray | torch.Tensor:
        preprocess = self.preprocess_sample(sample)
        part_nums = preprocess["part_nums"]
        shifted_start = preprocess["shifted_start"]
        pad_target_lens = preprocess["pad_target_lens"]
        # In this way, the same part of different input numbers will share the same shifted start.    
        numbers_tokens = sum(
            [(self.encode_number(
                s, 
                index_hint_shifted_start=shifted_start[:part_nums[i]] if shifted_start is not None else None,
                pad_target_lens = pad_target_lens[i] if pad_target_lens is not None else None,                     
            ) + [self.addition_token["split_token"]]) for i, s in enumerate(sample)],
            start = [])[:-1]
        question = [self.addition_token["bot_token"]] + numbers_tokens + [self.addition_token["answer_token"]]
        question = self.add_domain_and_task_prefix(tokens=question, task=task)
        

        def type_wrap(list_, return_tensor):
            if return_tensor == "no":
                return list_
            elif return_tensor == "pt":
                return torch.tensor(list_)
            elif return_tensor == "np":
                return np.array(list_)
            else:
                raise ValueError(
                    f"Invalid return_tensor value: {return_tensor}, support: ['no', 'pt', 'np']"
                )

        if not contain_answer:
            return type_wrap(question, return_tensor=return_tensor)
        else:
            return_ = (
                question
                + self.encode_number(self.get_answer(sample, task=task), index_hint=self.index_hint_in_answer)
                + [self.addition_token["eot_token"]]
            )
            return type_wrap(return_, return_tensor=return_tensor)

    @overload
    def pad_sample(self, encoded_sample: Sequence[int], batch_len: int, fillvalue: Number | None = None) -> list[int]: ...
    @overload
    def pad_sample(
        self, encoded_sample: torch.Tensor, batch_len: int, fillvalue: Number | None = None
    ) -> torch.Tensor: ...
    def pad_sample(
        self, encoded_sample: Sequence[int] | torch.Tensor, batch_len: int, fillvalue: Number | None = None
    ) -> list[int] | torch.Tensor:
        fillvalue: Number = fillvalue if fillvalue is not None else self.addition_token["pad_token"]
        if (
            (isinstance(encoded_sample, Sequence) and len(encoded_sample) > batch_len) 
            or (isinstance(encoded_sample, torch.Tensor) and encoded_sample.shape[0] > batch_len)
            ):
            return self.pad_sample(encoded_sample[-batch_len:], batch_len=batch_len, fillvalue=fillvalue)
        if self.padding_side == "left":
            if isinstance(encoded_sample, torch.Tensor):
                assert len(encoded_sample.shape) == 1
                assert encoded_sample.shape[0] <= batch_len
                return torch.cat(
                    (
                        fillvalue * torch.ones(
                            size=(batch_len - len(encoded_sample),),
                            dtype=encoded_sample.dtype,
                            device=encoded_sample.device,
                        ),
                        encoded_sample,
                    ),
                    dim=0,
                )
            else:
                assert len(encoded_sample) <= batch_len
                assert type(encoded_sample[0]) == type(fillvalue), f"provided fillvalue type should be the same as the encoded_sample type, but get {type(encoded_sample[0])} and {type(fillvalue)}."
                return [
                    fillvalue
                    for _ in range(batch_len - len(encoded_sample))
                ] + list(encoded_sample)
        elif self.padding_side == "right":
            if isinstance(encoded_sample, torch.Tensor):
                assert len(encoded_sample.shape) == 1
                assert encoded_sample.shape[0] <= batch_len
                return torch.cat(
                    (
                        encoded_sample,
                        fillvalue * torch.ones(
                            size=(batch_len - len(encoded_sample),),
                            dtype=encoded_sample.dtype,
                            device=encoded_sample.device,
                        ),
                    ),
                    dim=0,
                )
            else:
                assert len(encoded_sample) <= batch_len
                assert type(encoded_sample[0]) == type(fillvalue), f"provided fillvalue type should be the same as the encoded_sample type, but get {type(encoded_sample[0])} and {type(fillvalue)}."
                return list(encoded_sample) + [
                    fillvalue
                    for _ in range(batch_len - len(encoded_sample))
                ]
        else:
            raise ValueError(
                "Invalid padding_side {self.padding_side}. Should be 'left' or 'right'."
            )

    def encode_sample_batch(
        self, batch: Sequence[Sequence[NumberBasic]], contain_answer: bool = True
    ) -> torch.Tensor:
        encoded_batch: list[list[int]] = []
        for sample in batch:
            encoded_batch.append(
                self.encode_sample(sample, contain_answer=contain_answer)
            )
        max_length = max([len(sample) for sample in encoded_batch])
        for idx, encoded_sample in enumerate(encoded_batch):
            encoded_batch[idx] = self.pad_sample(
                encoded_sample=encoded_sample, batch_len=max_length
            )
        return torch.tensor(encoded_batch)

    def _check_token(self, token: str) -> None:
        """check whether a str is a valid token string."""
        if len(token) > self.digit:
            raise ValueError(f"Token {token} is too long.")
        if any(
            [
                (c not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
                for c in token
            ]
        ):
            raise ValueError(f"Token {token} contains non-digit characters.")

    def encode_one_token(self, int_: str) -> int:
        """Encode the number token to the token in the tokenizer."""
        # find the first non-zero digit
        # For each length, the token is from 00... to 10^length-1
        # For example, length 1: 0, 1, 2, ..., 9; length 2: 00, 01, ..., 99
        # So the token is the sum of 10^(d+1) for d in range(digit)
        self._check_token(int_)
        token_idx = sum(10**d for d in range(len(int_))) - 1 + int(int_) + self.num_addition_token
        assert (
            self.is_number_token(token_idx)
        ), f"The input {int_} cannot be converted into one token."
        return token_idx 

    def decode_one_token(
        self, int_: int | torch.Tensor | np.ndarray | np.integer
    ) -> str:
        """Decode the token (index) in the tokenizer to the string representation of ONE number token."""
        int_ = to_int(int_)
        if not self.is_number_token(int_):
            raise DecodeError(f"Token {int_} can not be convert to number.")
        int_ -= self.num_addition_token

        shorter_num = 10
        digit = 1
        while int_ >= shorter_num:
            int_ -= shorter_num
            digit += 1
            shorter_num *= 10
        return "0" * (digit - len(str(int_))) + str(int_)
    
    def add_index_hint(self, tokens: Sequence[int], count_side: Sequence[Literal["left", "right"]], hint_index: Sequence[Literal["left", "right"]], index_hint_shifted_start: Sequence[int] | None = None) -> list[int]:
        """tokens: the token idx of a NumberBasic instance. We add a hint token before each number token to show the digit of a token. A list of `hint` ("left" or "right") represents the index hint is based on the most-left digit or the most-right digit of a token. The sequence should have the same length of the number of part of the encoded number. For example, 1 for integer and 2 for float."""
        # split the token list into several list, by the element part token 
        parts = [list(group) for is_part_token, group in itertools.groupby(tokens, lambda x: x == self.addition_token["part_token"]) if not is_part_token]
        if not len(parts) == len(hint_index):
            raise ValueError(f"The length of input `hint_index` should be equal to the parts of the given nuber. Detect {len(parts)} parts and length of hint: {len(hint_index)}")
        if not len(parts) == len(count_side):
            raise ValueError(f"The length of input `count_side` should be equal to the parts of the given nuber. Detect {len(parts)} parts and length of hint: {len(count_side)}")
        if index_hint_shifted_start is not None:
            if len(index_hint_shifted_start) != len(parts):
                raise ValueError(f"The length of input `shifted_start` should be equal to the parts of the given nuber. Detect {len(parts)} parts and length of hint: {len(index_hint_shifted_start)}")
        else:
            index_hint_shifted_start: Sequence[int]  = [0 for _ in range(len(parts))]
        new_tokens: list[list[int]] = []
        for part_index, (part, cs, hi, start) in enumerate(zip(parts, count_side, hint_index, index_hint_shifted_start)):
            new_tokens.append(self._add_index_hint_for_part(tokens=part, count_side=cs, hint_index=hi, part_index=part_index, start=start))
            new_tokens.append([self.addition_token["part_token"]])
        return sum(new_tokens, start=[])[:-1] # remove the last part token
            
    def _add_index_hint_for_part(self, tokens: list[int], count_side: Literal["left", "right"], hint_index: Literal["left", "right"], part_index: int, start: int = 0) -> list[int]:
        """"Add a hint index before each number token (corresponding to the most left or right digit of the token)"""
        tokens = tokens[::-1] if count_side == "right" else tokens.copy()
        token_lens = [len(self.decode_one_token(t)) for t in tokens]
        
        assert count_side in ["left", "right"] and hint_index in ["left", "right"]
        if hint_index != count_side:
            # In this situation, the first hint index is len(t_0) - 1 + start.
            index = list(itertools.accumulate(token_lens, initial=-1 + start))[1:]
        else:
            # In this situation, the first hint index is 0.
            index = [start] + list(itertools.accumulate(token_lens, initial=start))[1:-1]
        hint_tokens = [self.get_hint_token(part_index=part_index, digit_index=i) for i in index]
        if count_side == 'right':
            hint_tokens = hint_tokens[::-1]
            tokens = tokens[::-1]
        return sum([[h,t] for h, t in zip(hint_tokens, tokens)], start=[])
    
    def remove_index_hint(self, tokens: list[int]) -> list[int]:
        return [t for t in tokens if not self.is_hint_token(t)]

    def visualize_hint_token(self, token: int) -> str:
        assert self.is_hint_token(token)
        assert self.max_index is not None
        part, digit = divmod(token - self.num_addition_token - self.num_number_token, self.max_index)
        if self.max_index <= 26:
            if part == 0:
                return chr(97 + digit)
            elif part == 1:
                return chr(65 + digit)
            elif part == 2:
                return chr(945 + digit)
            else:
                raise NotImplementedError("The hint token visualization is not implemented.")
        else:
            return f"<{chr(65+part)}{str(digit)}>"

    def visualize_token(
        self, token: int | np.integer | np.ndarray | torch.Tensor
    ) -> str:
        token = to_int(token)
        if self.is_addition_token(token):
            return self.addition_token_str[token]
        elif self.is_number_token(token):
            return self.decode_one_token(token)
        elif self.is_hint_token(token):
            return self.visualize_hint_token(token)
        elif self.is_domain_token(token):
            return self.domain_token_str[token]
        elif self.is_task_token(token):
            return self.task_token_str[token]
        else:
            raise ValueError(f'Token {token} cannot be visualized.')

    def visualize_sample(
        self, encoded_sample: Sequence[int] | np.ndarray | torch.Tensor
    ) -> str:
        """Only used to see the samples and debug. If you want to export a sample as natural language, use `export` instead."""
        encoded_sample = to_list(encoded_sample)
        return "".join([self.visualize_token(token) for token in encoded_sample])
    
    def export(self, encoded_sample: Sequence[int] | np.ndarray | torch.Tensor, digit_comma: bool = False) -> str:
        """Export a sample as natural language string to feed into other pre-defined tokenizers and pre-trained models. Compared to `visualize_sample`, an instruction of the task has been added at the begining of sample. The additional tokens are represented as the corresponding natural language strings instead of a `<xxx>`."""
        assert not self.multi_task, "Now export does not support for multi-tasks"
        assert self.task is not None
        tokens = to_list(encoded_sample)
        tokens_str = [self.visualize_token(token) for token in to_list(encoded_sample)]
        tokens_str = [tokens_str[i] + ("," if (digit_comma and i != (len(tokens_str) - 1) and self.is_number_token(tokens[i]) and self.is_number_token(tokens[i+1])) else "") for i in range(len(tokens_str))] 
        string = "".join(tokens_str)
        # remove all pad, bot, eot tokens if any
        string = string.replace(self.addition_token_str[self.addition_token["pad_token"]], "")
        string = string.replace(self.addition_token_str[self.addition_token["bot_token"]], "")
        string = string.replace(self.addition_token_str[self.addition_token["eot_token"]], "")
        # replace operator token into the task.operator
        if self.addition_token["split_token"] in encoded_sample:
            string = string.replace(self.addition_token_str[self.addition_token["split_token"]], self.task.operator())
        # replace answer token into the task.answer
        string = string.replace(self.addition_token_str[self.addition_token["answer_token"]], self.task.answer())
        # replace the part token into the sep_str for each number
        a_sep_strs = NumberBasic.get_subclass(self.task.input_a_type).sep_str()
        b_sep_strs = NumberBasic.get_subclass(self.task.input_b_type).sep_str() if self.task.input_b_type not in ["none", "int"] else []
        answer_sep_strs = NumberBasic.get_subclass(self.task.output_type).sep_str() if self.task.output_type != "int" else []
        sep_strs = ((a_sep_strs if a_sep_strs is not None else []) + 
                    (b_sep_strs if b_sep_strs is not None else []) + 
                    (answer_sep_strs if answer_sep_strs is not None else []))
        # the len of sep_strs should be equal to part tokens in string
        assert (c:=string.count(self.addition_token_str[self.addition_token["part_token"]])) == len(sep_strs), f"Inconsistent number of part tokens and sep_strs. Find {c} part tokens and get {len(sep_strs)} sep_strs from the task definition."
        # replace one by one
        sep_strs = sep_strs[::-1] if self.reverse_rep == "total" else sep_strs
        for sep_str in sep_strs:
            string = string.replace(self.addition_token_str[self.addition_token["part_token"]], sep_str, 1)
        # index hint tokens
        if self.index_hint:
            if self.export_index_hint_vocab is None:
                assert self.max_index <= 26, "Without providing `export_index_hint_vocab`, we use the default alphabet as the vocab. Therefore, `max_index` cannot be larger than 26."
                # In this situation, we can just use the index hint in visual
            else:
                assert self.max_index <= len(self.export_index_hint_vocab), f"The provided `export_index_hint_vocab` is shorter than setting `{MAX_PART_NUM} * max_index` {MAX_PART_NUM * self.max_index}."
                while (match := re.search(r"<(A|B|C|D|E|F|G)(\d+)>", string)) is not None:
                    string.replace(match.group(0), self._hint_token_to_vocab(match.group(0)))
        # add the instruction
        if self.task.task == "truediv" and self.task.input_a_type == "Fraction" and self.task.input_b_type == "Fraction":
            # we find all a/b / c/d and replace them as (a/b) / (c/d)
            string = re.sub(r"(\d+/\d+)\s*/\s*(\d+/\d+)", r"(\1) / (\2)", string)
        return self.task.instruction() + string
    
    def _hint_token_to_vocab(self, token: str) -> str:
        m = re.match(r"<(A|B|C|D|E|F|G)(\d+)>", token)
        if m is None:
            raise ValueError(f"Token {token} is not a hint token.")
        else:
            hint_part = ord(m.group(1)) - 65
            hint_index = int(m.group(2))
            return self.export_index_hint_vocab[hint_part * self.max_index + hint_index]

    def recover_input_number(
        self, encoded_sample: Sequence[int] | torch.Tensor | np.ndarray
    ) -> tuple[NumberBasic, NumberBasic] | NumberBasic | tuple[NumberBasic, int]:
        """Given the encoded sample, recover the input number(s)"""
        encoded_sample = to_list(encoded_sample)
        # remove pad tokens and eot token
        if self.multi_task:
            task = self.recover_task(encoded_sample)
            encoded_sample = encoded_sample[4:]
        else:
            assert self.task is not None, "If not multi-task, the `self.task` should be given when create the tokenizer."
            task = self.task
        assert task is not None
        encoded_sample = [
            token
            for token in encoded_sample
            if token
            not in [self.addition_token["bot_token"], self.addition_token["pad_token"], self.addition_token["eot_token"]]
        ]

        assert self.addition_token["answer_token"] in encoded_sample
        encoded_sample = encoded_sample[
            : encoded_sample.index(self.addition_token["answer_token"])
        ]

        if self.addition_token["split_token"] not in encoded_sample:
            """If the task is based on one input number, return the number directly."""
            assert task.input_b_type == "none"
            return self.decode_number(encoded_sample, expected_type=task.input_a_type) # type: ignore
        else:
            assert self.addition_token["split_token"] in encoded_sample
            split_index = encoded_sample.index(self.addition_token["split_token"])
            assert task.input_b_type != "none"
            return (
                self.decode_number(
                    encoded_sample[:split_index], expected_type=task.input_a_type),
                self.decode_number(
                    encoded_sample[split_index + 1 :], expected_type=task.input_b_type)
            ) # type: ignore
            
    def recover_task(self, encoded_sample: Sequence[int] | torch.Tensor | np.ndarray) -> Task | None:
        if not self.multi_task: return None
        encoded_sample = to_list(encoded_sample)
        assert all(map((lambda x: self.is_domain_token(x)), encoded_sample[:3])) and self.is_task_token(encoded_sample[3])
        task = Task(
            task = self.task_token_str[encoded_sample[3]],
            input_a_type = self.domain_token_str[encoded_sample[0]],
            input_b_type = self.domain_token_str[encoded_sample[1]],
            output_type = self.domain_token_str[encoded_sample[2]]
        )
        return task

    def retrieve_answer(
        self, encoded_sample: Sequence[int] | torch.Tensor | np.ndarray
    ) -> NumberBasic | int | None:
        """Retrieve the answer in the encoded input or output tokens."""
        encoded_sample = to_list(encoded_sample)
        if not self.addition_token["answer_token"] in encoded_sample:
            logger.warning(f"No answer token has been found in the encoded sample {encoded_sample}, no answer will be retrieved.")
            return None
        assert isinstance(encoded_sample, Sequence)
        encoded_sample = encoded_sample[
            encoded_sample.index(self.addition_token["answer_token"]) + 1 :
        ]
        # remove all padding tokens and eos tokens
        encoded_sample = [
            token for token in encoded_sample if (self.is_number_token(token) or token == self.addition_token["part_token"])
        ]
        if not encoded_sample:
            return None
        try:
            answer = self.decode_number(encoded_sample, expected_type=self.task.output_type)
        except DecodeError as e:
            logger.warning(f"Decode Error occurs when trying retrieving answer from seqeunce {encoded_sample}. Error message: {e}")
            return None
        return answer

    def pad(
        self,
        encoded_inputs: Sequence[torch.Tensor],
        padding: bool | str = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: Literal["pt", "np", "tf"] = "pt",
    ):
        if max_length is not None and max_length < 8:
            raise ValueError("The max_length should never smaller than 8.")
        if return_tensors != "pt":
            raise NotImplementedError("Only return_tensors='pt' is supported now.")
        if not padding:
            raise NotImplementedError("Unexpected behavior. Please check the pipeline.")
            return torch.stack(encoded_inputs, dim=0)
        batch_length = max([len(encoded_input) for encoded_input in encoded_inputs])
        if max_length is not None and batch_length > max_length:
            # warning about the exceeding
            index_ = ([len(encoded_input) for encoded_input in encoded_inputs]).index(
                batch_length
            )
            max_length_sample_str = str(encoded_inputs[index_])
            vis = self.visualize_sample(encoded_inputs[index_])
            logger.warning(
                "Batch length %d has exceed the max length %d, which could cause unexpected behavior. The longest sample in the batch: %s; visualization: %s",
                batch_length,
                max_length,
                max_length_sample_str,
                vis,
            )

            batch_length = min(batch_length, max_length)
        if pad_to_multiple_of is not None:
            raise NotImplementedError("pad_to_multiple_of is not supported now.")
        padded_inputs: list[torch.Tensor] = []
        padded_labels: list[torch.Tensor] = []
        attention_mask: list[torch.Tensor] = []
        position_ids: list[torch.Tensor] = []

        for encoded_input in encoded_inputs:
            padded_input = self.pad_sample(encoded_input, batch_len=batch_length)
            if self.random_position and self.np_rng.random() < self.random_position_prob:
                max_index = self.random_max_index
                min_index = self.random_min_index
                if (max_index - min_index) < (2 * batch_length):
                    logger.warning("The range of position_ids used in random_position is too small. Please set a larger range.")
                # random generate batch length integers in [0, 1024)
                pos = torch.tensor(
                        np.sort(
                            self.np_rng.choice(a=np.arange(min_index, max_index), size=encoded_input.shape[0], replace=False)
                        ), 
                        dtype=torch.int64, 
                        device=encoded_input.device
                        )
                position_ids.append(self.pad_sample(pos, batch_len=batch_length, fillvalue=0))
            elif self.random_shifted_start:
                start = self.np_rng.integers(0, 64) if self.np_rng.random() < self.random_shifted_prob else 0
                position_ids.append(
                    self.pad_sample(torch.arange(
                        start,
                        start + encoded_input.shape[0],
                        dtype=torch.int64,
                        device=encoded_input.device,
                    ), batch_len=batch_length, fillvalue=0)
                )
            else:
                pass
            labels = padded_input.clone()
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist() # type: ignore
            assert isinstance(labels, list)
            if self.addition_token["answer_token"] not in labels:
                logger.warning(
                    "The '=' token has not be included in the sample %s. It could cause an empty loss and cause unexpected behavior or raise error. Consider to enlarge the max_length or check your input."
                )
                labels = [-100 for _ in labels]
            else:
                labels[: labels.index(self.addition_token["answer_token"]) + 1] = [
                    -100
                    for _ in range(
                        labels.index(self.addition_token["answer_token"]) + 1
                    )
                ]
            labels = torch.tensor(
                labels, dtype=padded_input.dtype, device=padded_input.device
            )
            padded_inputs.append(padded_input)
            padded_labels.append(labels)

            max_length_ = max_length if max_length is not None else len(encoded_input)
            if self.padding_side == "right":
                
                attention_mask.append(
                    torch.tensor(
                        [1] * min(len(encoded_input), max_length_)
                        + [0] * (batch_length - len(encoded_input)),
                        dtype=torch.int64,
                        device=encoded_input.device,
                    )
                )
            elif self.padding_side == "left":
                attention_mask.append(
                    torch.tensor(
                        [0] * (batch_length - len(encoded_input))
                        + [1] * min(len(encoded_input), max_length_),
                        dtype=torch.int64,
                        device=encoded_input.device,
                    )
                )
            else:
                raise ValueError(
                    "Invalid padding_side {self.padding_side}. Should be 'left' or 'right'."
                )

        return {
            "input_ids": torch.stack(padded_inputs, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "position_ids": (
                torch.stack(position_ids, dim=0) if self.random_shifted_start or self.random_position else None
            ),
        }
