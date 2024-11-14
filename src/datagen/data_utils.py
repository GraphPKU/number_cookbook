import numpy as np
import torch
from typing import cast
from collections.abc import Sequence

def to_int(int_: int | np.integer | np.ndarray | torch.Tensor | float) -> int:
    """Convert a int-like object to int: int, np.integer, np.ndarray, torch.Tensor"""
    if isinstance(int_, np.ndarray):
        assert len(int_.shape) <= 1
        int_ = int_.item()
    elif isinstance(int_, np.integer):
        int_ = int_.item()
    elif isinstance(int_, torch.Tensor):
        assert len(int_.shape) <= 1
        int_ = cast(int, int_.cpu().detach().item())
    elif isinstance(int_, float):
        float_int = int_
        int_ = int(int_)
        if int_ != float_int:
            raise ValueError(f"Only support non-negative integers, but get a float: {float_int}")
    assert isinstance(int_, int), f"Only support int, np.integer, np.ndarray or torch.Tensor, but get a {type(int_)}: {int_}"
    return int_

def to_list(input_: int | Sequence[int] | np.integer | np.ndarray | torch.Tensor) -> list[int]:
    """Convert an object to a list of int: int, list[int], np.integer, np.ndarray, torch.Tensor"""
    check_int = (lambda list_like: all(map((lambda x: isinstance(x, int)), list_like)))
    if isinstance(input_, int):
        return [input_]
    elif isinstance(input_, Sequence) and check_int(input_):
        return list(input_)
    elif isinstance(input_, np.ndarray):
        assert len(input_.shape) <= 1
        if len(input_.shape) == 0:
            return [to_int(input_)]
        else:
            return_ = input_.tolist()
            assert check_int(return_)
            return return_
    elif isinstance(input_, np.integer):
        return [to_int(input_)]
    elif isinstance(input_, torch.Tensor):
        assert len(input_.shape) <= 1
        if len(input_.shape) == 0:
            return [to_int(input_)]
        else:
            return_ = input_.cpu().detach().tolist()
            assert check_int(return_)
            return return_
    else:
        raise ValueError(f'Invalid input {input_}')