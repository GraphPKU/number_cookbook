from typing import Literal, get_args

ReverseType = Literal["int", "dec", "each", "total", "no"]
REVERSE_CHOICES = get_args(ReverseType)