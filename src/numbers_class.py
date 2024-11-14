from abc import ABC, ABCMeta, abstractmethod
from typing import Literal, cast, TypeVar, Generic, Self, get_args
import itertools
import math
import random

# Domain = Literal["Integer", "Float", "Fraction", "ScientificNotation", "Vector"]
Domain = Literal["Integer", "Float", "Fraction", "ScientificNotation"]
DOMAIN_CHOICES = get_args(Domain)
MAX_PART_NUM = 3 # The maximum number of parts in a number, the largest is 3 in scientific notation
    
class NumberBasic(ABC):
    ...
    
    @abstractmethod
    def to_string(self) -> str:
        ...
        
    def __str__(self) -> str:
        return self.to_string()
    
    @property
    def value(self):
        """The value of the number could not be accurate."""
        assert hasattr(self, '_value'), "The value attribute is not set"
        return self._value

    @classmethod
    @abstractmethod
    def from_string(cls, value: str) -> Self:
        ...
        
    def __ne__(self, value: object) -> bool:
        return not self == value
    
    @abstractmethod
    def __gt__(self, other: Self) -> bool:
        ...
        
    @abstractmethod
    def __lt__(self, other: Self) -> bool:
        ...
    
    def __ge__(self, other: Self) -> bool:
        return self == other or self > other
    
    def __le__(self, other: Self) -> bool:
        return self == other or self < other
    
    @classmethod
    def get_subclass(cls, domain: Domain) -> type['NumberBasic']:
        return globals()[domain]
    
    def __hash__(self):
        return hash(self.to_string())
    
    def hash(self):
        return self.__hash__()
    
    
    @classmethod
    @abstractmethod
    def sep_str(cls) -> None | list[str]:
        ...
        
    @property
    def num_part(self) -> int:
        sep_str = self.sep_str()
        if sep_str is None:
            return 1
        else:
            return len(sep_str) + 1
        
    @property
    def sep_index(self) -> None | list[int]:
        sep_str = self.sep_str()
        if sep_str is None:
            return None
        return_list: list[int] = []
        str_ = str(self)
        for si in sep_str:
            return_list.append(str_.index(si))
            str_ = str_[str_.index(si)+1:]
        return return_list
    
    def part_str(self) -> list[str]:
        if self.sep_index is None:
            return [str(self)]
        return_list = []
        str_ = str(self)
        for si in self.sep_index:
            return_list.append(str_[:si])
            str_ = str_[si+1:]
        return_list.append(str_)
        assert len(return_list) == self.num_part
        return return_list
    
    @classmethod
    @abstractmethod
    def default_read_direct(cls) -> list[Literal['left', 'right']]:
        ...
    
    @classmethod
    @abstractmethod
    def random_generate(cls, length: int, *, random_rng: random.Random | None = None) -> Self:
        ...
        
    @property
    @abstractmethod
    def digit(self) -> int:
        ...
        
    @property
    @abstractmethod
    def digit_part(self) -> list[int]:
        ...
        
    @abstractmethod
    def __add__(self, other: Self) -> Self:
        ...
        
    @abstractmethod
    def __sub__(self, other: Self) -> Self:
        ...
        
    @abstractmethod
    def __mul__(self, other: Self) -> Self:
        ...
        
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...
        
    def trunc_digit(self, target_digit: int) -> Self:
        if self.digit <= target_digit:
            return self
        return self._trunc_digit(target_digit)
    
    def _trunc_digit_int(self, target_digit: int, int_: int) -> int:
        return int_ % (10 ** target_digit)
    
    @abstractmethod
    def _trunc_digit(self, target_digit: int) -> Self:
        ...
    
    @classmethod
    def get_pattern(cls, domain: Domain | Literal["int"]) -> str:
        if domain == "Integer" or domain == "int":
            return r"\d+"
        if domain == "Float":
            return r"\d+\.\d+"
        if domain == "Fraction":
            return r"\d+/\d+"
        if domain == "ScientificNotation":
            return r"\d+\.\d+[eE]\d+"
        raise ValueError(f"Invalid domain {domain}")
        
    
class Integer(NumberBasic):
    def __init__(self, value: int):
        assert isinstance(value, int), f"The value should be an integer but find {value}"
        self._value = value
        
    def to_string(self) -> str:
        return str(self.value)
    
    @classmethod
    def from_string(cls, value: str) -> Self:
        return cls(int(value))
    
    @classmethod
    def from_int(cls, value: int) -> Self:
        return cls(value)
    
    def __add__(self, other: Self) -> Self:
        return type(self)(self.value + other.value)
    
    def __sub__(self, other: Self) -> Self:
        if self < other:
            raise NotImplementedError('Now we do not support negative integer.')
        return type(self)(self.value - other.value)
    
    def __floordiv__(self, other: Self) -> Self:
        return type(self)(self.value // other.value)
    
    def __mod__(self, other: Self) -> Self:
        return type(self)(self.value % other.value)
    
    def __mul__(self, other: Self) -> Self:
        return type(self)(self.value * other.value)
    
    def __truediv__(self, other: Self) -> "Fraction":
        if other.value == 0:
            raise ZeroDivisionError("Division by zero.")
        return Fraction(self.value, other.value)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Integer):
            return False
        return self.value == other.value
    
    def __gt__(self, other: Self) -> bool:
        return self.value > other.value
    
    def __lt__(self, other: Self) -> bool:
        return self.value < other.value
    
    @classmethod
    def default_read_direct(cls) -> list[Literal['left', 'right']]:
        return ["right"]
    
    @classmethod
    def random_generate(cls, length: int, *, random_rng: random.Random | None = None) -> Self:
        assert length >= 1
        min_ = 10 ** (length - 1) if length != 1 else 0
        if random_rng is None:
            random_rng = random.Random()
        return cls(random_rng.randint(min_, 10 ** length - 1))
    
    @property
    def digit(self):
        return len(str(self.value))
    
    @property
    def digit_part(self):
        return [self.digit]
    
    def __hash__(self):
        return super().__hash__()
    
    @classmethod
    def sep_str(cls):
        return None
    
    def _trunc_digit(self, target_digit: int) -> Self:
        return type(self)(self._trunc_digit_int(target_digit, self.value))
    
    def to_scient(self, sig: int | None = None) -> 'ScientificNotation':
        return Float(self.value, [0]).to_scient(sig=sig)
        
    
class Float(NumberBasic):
    def __init__(self, int_part: int, dec_part: list[int], *, keep_zeros: int = 0):
        """
        int_part: a python int;
        dec_part: a list with element in [0-9].
        """
        assert isinstance(int_part, int), f"The integer part should be an integer but find {int_part}"
        self.int_part = int_part
        self.keep_zeros = keep_zeros
        # remove zeros from the right in the decimal part
        assert isinstance(dec_part, list), "The decimal part should be a list"
        assert all(digit in list(range(10)) for digit in dec_part), "The decimal part should contain only digits"
        end_zeros = 0
        while dec_part and dec_part[-1] == 0:
            if keep_zeros > 0:
                keep_zeros -= 1
                end_zeros += 1
            dec_part.pop()
        if not dec_part:
            dec_part.append(0)
        dec_part = dec_part + [0] * end_zeros
        self.dec_part = dec_part
        try:
            self._value = float(self.to_string())
        except OverflowError as e:
            self._value = float('inf')
        
    def _str_for_dec_part(self, dec_part: list[int]) -> str:
        return "".join(map(str, dec_part))
        
    def to_string(self):
        return f"{self.int_part}.{self._str_for_dec_part(self.dec_part)}"
    
    def __add__(self, other: Self) -> Self:
        int_part = self.int_part + other.int_part
        # calculate the decimal part
        pad_decimal = [(a, b) for a, b in itertools.zip_longest(self.dec_part, other.dec_part, fillvalue=0)]
        results: list[int] = []
        carry: int = 0
        for a, b in pad_decimal[::-1]:
            results.append((a+b+carry) % 10)
            carry = (a + b + carry) // 10
        results.reverse()
        if carry:
            int_part += carry
        return type(self)(int_part, results)
    
    def __sub__(self, other: Self) -> Self:
        if self < other:
            raise NotImplementedError('Now we do not support negative float.')
        int_part = self.int_part - other.int_part
        # calculate the decimal part
        pad_decimal = [(a, b) for a, b in itertools.zip_longest(self.dec_part, other.dec_part, fillvalue=0)]
        results: list[int] = []
        borrow: int = 0
        for a, b in pad_decimal[::-1]:
            results.append((a-b-borrow) % 10)
            borrow = 1 if (a - b - borrow) < 0 else 0
        results.reverse()
        if borrow:
            int_part -= 1
        return type(self)(int_part, results)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Float):
            return False
        return self.int_part == other.int_part and self.dec_part == other.dec_part
    
    def __hash__(self):
        return super().__hash__()
    
    def __gt__(self, other: Self) -> bool:
        if self.int_part > other.int_part:
            return True
        if self.int_part < other.int_part:
            return False
        for a, b in itertools.zip_longest(self.dec_part, other.dec_part, fillvalue=0):
            if a > b:
                return True
            if a < b:
                return False
        return False
    
    def __lt__(self, other: Self) -> bool:
        if self.int_part < other.int_part:
            return True
        if self.int_part > other.int_part:
            return False
        for a, b in itertools.zip_longest(self.dec_part, other.dec_part, fillvalue=0):
            if a < b:
                return True
            if a > b:
                return False
        return False
    
    def to_int_rep(self) -> tuple[int, int]:
        """
        Translate a Float into two int, where the float itself should be equal to int_1 * 10 ** (- int_2).
        Notice that int_2 always >= 1. So the Float(1, [0]) will be into (10, 1).
        """
        dec_part = self._str_for_dec_part(self.dec_part)
        return int(str(self.int_part) + dec_part), len(dec_part)
    
    @classmethod
    def from_int_rep(cls, value: int, dec_len: int, *, keep_zeros: int = 0) -> Self:
        assert dec_len >= 0
        int_part = value // (10 ** dec_len)
        if dec_len == 0:
            dec_string = ""
        else:
            dec_string = str(value)[-dec_len:]
            dec_string = "0" * (dec_len - len(dec_string)) + dec_string
        
        dec_part = [int(digit) for digit in dec_string]
        return cls(int_part, dec_part, keep_zeros=keep_zeros)
    
    def __mul__(self, other: Self) -> Self:
        self_int, self_len = self.to_int_rep()
        other_int, other_len = other.to_int_rep()
        return self.from_int_rep(self_int * other_int, self_len + other_len)
            
    @classmethod
    def from_string(cls, value: str) -> Self:
        if '.' not in value:
            return cls(int(value), [0])
        int_part, dec_part = value.split('.')
        return cls(int(int_part), [int(digit) for digit in dec_part if int(digit) in list(range(10))])
    
    @classmethod
    def from_float(cls, value: float) -> Self:
        assert value >= 0
        value = value * 1.0
        return cls.from_string(str(value))
    
    @classmethod
    def default_read_direct(cls) -> list[Literal['left', 'right']]:
        return ["right", "left"]
    
    @classmethod
    def random_generate(cls, length: int, *, random_rng: random.Random | None = None) -> Self:
        assert length >= 1
        if random_rng is None:
            random_rng = random.Random()
        int_digit = random_rng.randint(1, length)
        int_part = random_rng.randint(10 ** (int_digit - 1), 10 ** int_digit - 1)
        # int_part = random_rng.randint(0, 10 ** length - 1)
        if length >= 1:
            dec_part = [random_rng.randint(0, 9) for i in range(length-1)] + [random_rng.randint(1,9)]
        else:
            dec_part = [random_rng.randint(0,9)] # If there is only one digit, the last digit of decimal part can be zero
        return cls(int_part, dec_part)
    
    @property
    def digit(self):
        return len(self.dec_part)
    
    @property
    def digit_part(self):
        return [len(str(self.int_part)), len(self.dec_part)] 
    
    @classmethod
    def sep_str(cls):
        return ['.']
    
    def _trunc_digit(self, target_digit: int) -> Self:
        return type(self)(self._trunc_digit_int(target_digit, self.int_part), self.dec_part[:target_digit])
    
    def to_scient(self, sig: int | None = None) -> 'ScientificNotation':
        if sig is None:
            return ScientificNotation(self, 0)
        # convert to scientific notation and keep sig significant digits
        assert sig >= 1, f"The significant digits should be positive but find {sig}."
        int_rep_value, dec_len = self.to_int_rep()
        new_int = int(str(int_rep_value)[:sig])
        if len(str(int_rep_value)) <= sig: # the sig. fig. is the last digit 
            pass
        elif int(str(int_rep_value)[sig]) > 5:
            new_int += 1
        elif int(str(int_rep_value)[sig] == 5) and int(str(int_rep_value)[sig - 1]) % 2 == 1:
            new_int += 1
            
        # calculate how many zeros at the end of new_int, these zeros should be kept in the result
        end_zeros = len(str(new_int)) - len(str(new_int).rstrip('0')) if new_int != 0 else 0
            
        new_int = new_int * 10 ** (len(str(int_rep_value)) - len(str(new_int)))
        return ScientificNotation(Float.from_int_rep(new_int, dec_len, keep_zeros=end_zeros), 0)
    
    def _times_10(self, times: int = 1) -> Self:
        if times == 0:
            return self
        assert times > 0, f"The times should be positive but find {times}."
        dec_to_int = int(self._str_for_dec_part(self.dec_part)[:times])
        int_ = self.int_part * 10 ** times + dec_to_int * 10 ** (times - min(times, len(self.dec_part)))
        dec_ = self.dec_part[times:]
        return type(self)(int_, dec_)
    
class Fraction(NumberBasic):
    def __init__(self, num: int, den: int):
        num, den = self._reduce(num, den)
        self.num = num
        self.den = den
        try:
            self._value = self.num / self.den
        except OverflowError as e:
            self._value = float('inf')
        
    def to_string(self):
        return f"{self.num}/{self.den}"
    
    def __add__(self, other: Self) -> Self:
        num = self.num * other.den + other.num * self.den
        den = self.den * other.den
        return type(self)(num, den)
    
    def __sub__(self, other: Self) -> Self:
        if self < other:
            raise NotImplementedError('Now we do not support negative fraction.')
        num = self.num * other.den - other.num * self.den
        den = self.den * other.den
        return type(self)(num, den)
    
    @staticmethod
    def _reduce(num: int, den: int) -> tuple[int, int]:
        common = math.gcd(num, den)
        return num // common, den // common
    
    @classmethod
    def from_string(cls, value: str) -> Self:
        num, den = map(int, value.split('/'))
        return cls(num, den)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fraction):
            return False
        return self.num * other.den == other.num * self.den
    
    def __hash__(self):
        return super().__hash__()
    
    def __gt__(self, other: Self) -> bool:
        return self.num * other.den > other.num * self.den
    
    def __lt__(self, other: Self) -> bool:
        return self.num * other.den < other.num * self.den
    
    def __mul__(self, other: Self) -> Self:
        return type(self)(self.num * other.num, self.den * other.den)
    
    def __truediv__(self, other: Self) -> Self:
        return type(self)(self.num * other.den, self.den * other.num)
    
    @classmethod
    def default_read_direct(cls) -> list[Literal['left', 'right']]:
        return ["right", "right"]
    
    @classmethod
    def random_generate(cls, length: int, *, random_rng: random.Random | None = None) -> Self:
        """length: the number of digits in the denominator"""
        assert length >= 1
        if random_rng is None:
            random_rng = random.Random()
        while True:
            num = random_rng.randint(0, 10 ** length - 1)
            den = random_rng.randint(10 ** (length - 1), 10 ** length - 1)
            red_num, red_den = cls._reduce(num, den)
            if len(str(red_den)) == length:
                break
        return cls(red_num, red_den)
    
    @property
    def digit(self):
        return max(len(str(self.num)), len(str(self.den)))
    
    @property
    def digit_part(self):
        return [len(str(self.num)), len(str(self.den))]
    
    @classmethod
    def sep_str(cls):
        return ['/']
    
    def to_float(self):
        # if the fraction cannot be represented as finite float, raise an error
        # check whether the den has factor other than 2 or 5
        factor_2 = 0
        factor_5 = 0
        den = self.den
        while den % 2 == 0:
            den //= 2
            factor_2 += 1
        while den % 5 == 0:
            den //= 5
            factor_5 += 1
        if den != 1:
            raise ValueError("The fraction cannot be represented as finite float.")
        return Float.from_int_rep(
            self.num * (2 ** factor_5) * (5 ** factor_2),
            factor_2 + factor_5
        )
        
    def _trunc_digit(self, target_digit: int) -> Self:
        den = self._trunc_digit_int(target_digit=target_digit, int_=self.den)
        if den == 0:
            den = 1
        num = self._trunc_digit_int(target_digit=target_digit, int_=self.num)
        return type(self)(num, den)
    
class ScientificNotation(NumberBasic):
    random_exponent_range = (0,99)
    def __init__(self, mantissa: Float, exponent: int):
        assert isinstance(mantissa, Float), f"The mantissa should be a Float but get {type(mantissa)} {mantissa}"
        assert isinstance(exponent, int), f"The exponent should be an integer but get {type(exponent)} {exponent}"
        self.mantissa, self.exponent = self._reduce(mantissa, exponent)
        if self.exponent < 0:
            raise ValueError(f"The exponent should be non-negative for now, but find mantissa {mantissa} and exponent {exponent}.")
        try:
            if self.exponent > 99:
                raise OverflowError(f"The exponent should be less than 100 but get {self.exponent}.")
            self._value = self.mantissa.value * 10 ** self.exponent
        except OverflowError as e:
            self._value = float('inf')
        
    def _reduce(self, mantissa: Float, exponent: int) -> tuple[Float, int]:
        if mantissa == Float(0, [0]):
            return mantissa, 0
        if mantissa.int_part == 0:
            new_int_part = mantissa.dec_part[0]
            new_dec_part = mantissa.dec_part[1:]
            new_exponent = exponent - 1
            return self._reduce(Float(new_int_part, new_dec_part, keep_zeros=mantissa.keep_zeros), new_exponent)
        if mantissa.int_part >= 10:
            new_int_part = mantissa.int_part // 10
            new_dec_part = [mantissa.int_part % 10] + mantissa.dec_part
            new_exponent = exponent + 1
            return self._reduce(Float(new_int_part, new_dec_part, keep_zeros=mantissa.keep_zeros), new_exponent)
        if exponent < 0:
            raise ValueError(f"The exponent should be non-negative for now, but find mantissa {mantissa} and exponent {exponent}.")
        return mantissa, exponent
        
    def to_string(self):
        return f"{self.mantissa.to_string()}e{self.exponent}"
    
    def to_float(self):
        if self.exponent >= 0:
            return self.mantissa._times_10(self.exponent)
        else:
            exp_float = Float(0, [0] * (-self.exponent - 1) + [1])
            return self.mantissa * exp_float
    
    def __add__(self, other: Self) -> Self:
        if self.exponent < other.exponent:
            return other + self
        mantissa = self.mantissa * Float(int_part=(10 ** (self.exponent - other.exponent)), dec_part=[0]) + other.mantissa
        return type(self)(mantissa, other.exponent)
    
    def __sub__(self, other: Self) -> Self:
        if self.exponent < other.exponent:
            raise NotImplementedError('Now we do not support negative scientific notation.')
        mantissa = self.mantissa * Float(int_part=(10 ** (self.exponent - other.exponent)), dec_part=[0]) - other.mantissa
        return type(self)(mantissa, other.exponent)
    
    def __mul__(self, other: Self) -> Self:
        man = self.mantissa * other.mantissa
        exp = self.exponent + other.exponent
        return type(self)(man, exp)
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScientificNotation):
            return False
        return self.mantissa == other.mantissa and self.exponent == other.exponent
    
    def __hash__(self):
        return super().__hash__()
    
    def __gt__(self, other: Self) -> bool:
        if self.exponent > other.exponent:
            return True
        if self.exponent < other.exponent:
            return False
        return self.mantissa > other.mantissa
    
    def __lt__(self, other: Self) -> bool:
        if self.exponent < other.exponent:
            return True
        if self.exponent > other.exponent:
            return False
        return self.mantissa < other.mantissa
    
    @classmethod
    def from_string(cls, value: str) -> Self:
        mantissa, exponent = value.split('e')
        return cls(Float.from_string(mantissa), int(exponent))
    
    @classmethod
    def default_read_direct(cls) -> list[Literal['left', 'right']]:
        return ["right", "left", "right"]
    
    @classmethod
    def random_generate(cls, length: int, *, random_rng: random.Random | None = None) -> Self:
        if random_rng is None:
            random_rng = random.Random()
        float_int = random_rng.randint(1, 9)
        float_dec = Float.random_generate(length, random_rng=random_rng).dec_part
        float_part = Float(float_int, float_dec)
        return cls(float_part, random_rng.randint(cls.random_exponent_range[0], cls.random_exponent_range[1]))
    
    @property
    def digit(self):
        return self.mantissa.digit
    
    @property
    def digit_part(self):
        return self.mantissa.digit_part + [len(str(self.exponent))]
    
    @classmethod
    def sep_str(cls):
        return ['.', 'e']
    
    def _trunc_digit(self, target_digit: int) -> Self:
        # TODO: negative
        man = self.mantissa.trunc_digit(target_digit)
        if man < Float(1, [0]):
            man = man + Float(1, [0])
        return type(self)(man, self.exponent)
    
# class Vector(NumberBasic):
#     def __init__(self, values: list[NumberBasic]):
#         self._values = values
#         raise NotImplementedError
        
#     def to_string(self) -> str:
#         return "[" + ", ".join(value.to_string() for value in self._values) + "]"

#     @classmethod
#     def from_string(cls, value: str, element_type: str) -> 'Vector':
#         values = value[1:-1].split(', ')
#         Type = globals()[element_type]
#         # Type should be a subclass of NumberBasic
#         assert issubclass(Type, NumberBasic), "Type should be a subclass of NumberBasic."
#         return cls([Type.from_string(value) for value in values])
    
#     ...