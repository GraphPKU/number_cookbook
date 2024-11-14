from numbers_class import Domain, NumberBasic, Integer, Float, Fraction, ScientificNotation, DOMAIN_CHOICES
from typing import get_args, Literal, Callable, Sequence, TypeVar, Self
import itertools
import random

TaskType = Literal["add", "add_easy", "sub", "max", "max_hard", "min", "min_hard", "multiply_hard", "multiply_easy", "digit_max", "digit_min", "digit_add", "get_digit", "length", "truediv", "floordiv", "mod", "mod_easy", "to_float", "to_scient", "count", "sig"]
TASK_CHOICES = get_args(TaskType)
TASKNUM = len(get_args(TaskType))

T = TypeVar("T", bound=NumberBasic)
MULTIPLY_EASY_MAX_DIGIT = 3
ADD_EASY_MAX_DIGIT = 3
MOD_EASY_MAX_DIGIT = 3
SCIENT_EXP_DIFF_RANGE = 5

class Task:
    all_tasks: list[str] = list(TASK_CHOICES)
    valid_tasks: list[tuple[str, Domain, Domain | Literal["none", "int"], Domain | Literal["int"]]] = [
        ("add", "Integer", "Integer", "Integer"),
        ("add", "Float", "Float", "Float"),
        ("add", "Fraction", "Fraction", "Fraction"),
        ("add_easy", "Fraction", "Fraction", "Fraction"), # preprocess: the digit of one fraction <= 3
        ("add", "ScientificNotation", "ScientificNotation", "ScientificNotation"), # preprocess: diff_exp <= 5
        
        ("sub", "Integer", "Integer", "Integer"),
        ("sub", "Float", "Float", "Float"),
        ("sub", "Fraction", "Fraction", "Fraction"),
        ("sub", "ScientificNotation", "ScientificNotation", "ScientificNotation"),
        
        ("max", "Integer", "Integer", "Integer"),
        ("max", "Float", "Float", "Float"),
        ("max", "Fraction", "Fraction", "Fraction"), # preprocess: frac < 1 
        ("max", "ScientificNotation", "ScientificNotation", "ScientificNotation"),
        
        ("max_hard", "Integer", "Integer", "Integer"),
        ("max_hard", "Float", "Float", "Float"),
        ("max_hard", "ScientificNotation", "ScientificNotation", "ScientificNotation"), # Do the same thing as max. Only to denote the task is hard about dataset.
        
        ("min", "Integer", "Integer", "Integer"),
        ("min", "Float", "Float", "Float"),
        ("min", "Fraction", "Fraction", "Fraction"), # preprocess: frac < 1 
        ("min", "ScientificNotation", "ScientificNotation", "ScientificNotation"),
        
        ("min_hard", "Integer", "Integer", "Integer"),
        ("min_hard", "Float", "Float", "Float"),
        ("min_hard", "ScientificNotation", "ScientificNotation", "ScientificNotation"), # Do the same thing as min. Only to denote the task is hard about dataset.
        
        ("multiply_hard", "Integer", "Integer", "Integer"),
        ("multiply_hard", "Float", "Float", "Float"),
        ("multiply_hard", "Fraction", "Fraction", "Fraction"),
        ("multiply_hard", "ScientificNotation", "ScientificNotation", "ScientificNotation"), # In fact, we don't preprocess for m_hard, ``hard`` should be in the dataset. (`test_shorter_len` == 0.5)
        
        ("multiply_easy", "Integer", "Integer", "Integer"),
        ("multiply_easy", "Float", "Float", "Float"),
        ("multiply_easy", "Fraction", "Fraction", "Fraction"),
        ("multiply_easy", "ScientificNotation", "ScientificNotation", "ScientificNotation"), # make the b is shorter than a fixed value. 
        
        ("digit_max", "Integer", "Integer", "Integer"),
        ("digit_max", "Float", "Float", "Float"),
        
        ("digit_min", "Integer", "Integer", "Integer"),
        ("digit_min", "Float", "Float", "Float"),
        
        ("digit_add", "Integer", "Integer", "Integer"),
        ("digit_add", "Float", "Float", "Float"),
        
        ("get_digit", "Integer", "int", "int"),
        ("get_digit", "Float", "int", "int"),
        
        ("length", "Integer", "none", "int"),
        ("length", "Float", "none", "int"),
        
        ("truediv", "Integer", "Integer", "Fraction"),
        ("truediv", "Fraction", "Fraction", "Fraction"),
        
        ("floordiv", "Integer", "Integer", "Integer"),
        ("mod", "Integer", "Integer", "Integer"),
        ("mod_easy", "Integer", "Integer", "Integer"),
        
        ("to_float", "Fraction", "none", "Float"),
        ("to_float", "ScientificNotation", "none", "Float"),
        
        ("to_scient", "Integer", "none", "ScientificNotation"), # preprocess: more zero
        ("to_scient", "Float", "none", "ScientificNotation"), # preprocess: more zero
        
        ("count", "Integer", "int", "int"),
        
        ("sig", "Integer", "int", "ScientificNotation"),
        ("sig", "Float", "int", "ScientificNotation"),
    ]

    def __init__(self, task: str, input_a_type: Domain, input_b_type: Domain | Literal["none", "int"], output_type: Domain | Literal["int"]):
        assert task in self.all_tasks, "Invalid task {} is not in {}".format(task, self.all_tasks)
        self.task = task

        if input_a_type not in DOMAIN_CHOICES:
            raise ValueError(f"Invalid input_a_type {input_a_type}")
        if input_b_type not in ["none", "int"] and input_b_type not in DOMAIN_CHOICES:
            raise ValueError(f"Invalid input_b_type {input_b_type}")
        self.input_a_type: Domain = input_a_type
        self.input_b_type: Domain | Literal["int", "none"] = input_b_type
        if output_type != "int" and output_type not in DOMAIN_CHOICES:
            raise ValueError(f"Invalid output_type {output_type}")
        self.output_type: Domain | Literal["int"] = output_type
        
        self._check_valid_task(task, input_a_type, input_b_type, output_type)
        
    def name(self) -> str:
        """Name of a task, which is task_title_task_input_a_type_input_b_type_output_type"""
        return "_".join([self.task, self.input_a_type, self.input_b_type, self.output_type])
        
    @classmethod
    def name2components(cls, name: str) -> tuple[str, Domain, Domain | Literal["none", "int"], Domain | Literal["int"]]:
        """Split the name of a task to its (title, input_a_type, input_b_type, output_type) components."""
        c = name.split("_")
        comp =  "_".join(c[:-3]), c[-3], c[-2], c[-1]
        comp = cls._check_valid_task(*comp)
        return comp
        
    @classmethod
    def _check_valid_task(cls, task: str, input_a_type: str, input_b_type: str, output_type: str) -> tuple[str, Domain, Domain | Literal["none", "int"], Domain | Literal["int"]]:
        if input_a_type not in DOMAIN_CHOICES:
            raise ValueError(f"Invalid input_a_type {input_a_type}")
        if input_b_type not in ["none", "int"] and input_b_type not in DOMAIN_CHOICES:
            raise ValueError(f"Invalid input_b_type {input_b_type}")
        if output_type != "int" and output_type not in DOMAIN_CHOICES:
            raise ValueError(f"Invalid output_type {output_type}")
        if (task, input_a_type, input_b_type, output_type) not in cls.valid_tasks:
            raise ValueError(f"Invalid task {task} with input_a_type {input_a_type}, input_b_type {input_b_type}, output_type {output_type}.")
        return task, input_a_type, input_b_type, output_type # type: ignore
    
    @classmethod
    def auto_task(cls, task: str, input_a_type: Domain) -> Self:
        """Because the input_b_type and output_type are conditional on the task and input_a_type, we can use this method to automatically generate a Task object."""
        for t in cls.valid_tasks:
            if t[0] == task and t[1] == input_a_type:
                return cls(*t)
        else:
            raise ValueError(f"Invalid task {task} with input_a_type {input_a_type}. Please check the valid tasks using `task.valid_tasks`.")
        
    def __str__(self):
        return f"Task({self.task}, {self.input_a_type}, {self.input_b_type}, {self.output_type})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return False
        return str(self) == str(other)
        
    def preprocess_data(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic | int | None]:
        """
        Preprocess the input data (pair of numbers) for the task.
        """
        assert hasattr(self, f"_preprocess_{self.task}"), f"Method preprocess_{self.task} not found."
        a, b = getattr(self, f"_preprocess_{self.task}")(input_numbers)
        assert type(a).__name__ == self.input_a_type, f"First input type {type(a).__name__} inconsistent with task define {self.input_a_type}"
        if b is None:
            assert self.input_b_type == "none", f"Second input must be None if task define the 'input_b_type' as 'none'"
        else:
            assert type(b).__name__ == self.input_b_type, f"Second input type {type(b).__name__} inconsistent with task define {self.input_b_type}"
        return a, b
    
    def _preprocess_same_type(self, input_numbers: Sequence[NumberBasic | int], task_name: str) -> tuple[NumberBasic, NumberBasic]:
        a, b = input_numbers[0], input_numbers[1]
        assert type(a) == type(b)
        assert isinstance(a, NumberBasic) and isinstance(b, NumberBasic), f"Input type {type(a)} and {type(b)} is not supported for task {task_name}."
        return a, b
    
    def _preprocess_larger_digit_first(self, input_numbers: Sequence[NumberBasic | int], task_name: str) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_same_type(input_numbers, task_name)
        return (a, b) if (a.digit >= b.digit) else (b, a)
    
    def _preprocess_larger_first(self, input_numbers: Sequence[NumberBasic | int], task_name: str) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_same_type(input_numbers, task_name)
        return (a, b) if (a >= b) else (b, a)
    
    def _preprocess_add(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_same_type(input_numbers, "Add")
        if isinstance(a, ScientificNotation) and isinstance(b, ScientificNotation):
            # make the exponent difference less than 5
            exp_upper_bound = ScientificNotation.random_exponent_range[1]
            exp_lower_bound = ScientificNotation.random_exponent_range[0]
            if a.exponent == exp_upper_bound:
                a = type(a)(a.mantissa, a.exponent - 1)
            if b.exponent == exp_upper_bound:
                b = type(b)(b.mantissa, b.exponent - 1)
            exp_diff = self._tmp_random_rng(seed=int(b.part_str()[0])).randint(-SCIENT_EXP_DIFF_RANGE, SCIENT_EXP_DIFF_RANGE)
            b = type(b)(b.mantissa, max(min(a.exponent + exp_diff, exp_upper_bound - 1), exp_lower_bound))
        return a, b
    
    def _preprocess_add_easy(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_add(input_numbers)
        assert isinstance(a, Fraction) and isinstance(b, Fraction), "Add_easy task only support two Fraction inputs."
        if a.digit > b.digit:
            b = b._trunc_digit(min(b.digit, ADD_EASY_MAX_DIGIT))
        else:
            a = a._trunc_digit(min(a.digit, ADD_EASY_MAX_DIGIT))
        return a, b
        
    def _preprocess_sub(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_same_type(input_numbers, "Sub")
        if b > a:
            a, b = b, a
            
        if isinstance(a, ScientificNotation) and isinstance(b, ScientificNotation):
            exp_upper_bound = ScientificNotation.random_exponent_range[1]
            exp_lower_bound = ScientificNotation.random_exponent_range[0]
            if a.exponent == exp_lower_bound:
                a = type(a)(a.mantissa, a.exponent + 1)
            # make the exponent difference less than 5
            exp_diff = self._tmp_random_rng(seed=int(b.part_str()[0])).randint(-SCIENT_EXP_DIFF_RANGE, -1)
            b = type(b)(b.mantissa, max(min(a.exponent + exp_diff, exp_upper_bound), exp_lower_bound))
        
        if isinstance(a, ScientificNotation) and isinstance(b, ScientificNotation) and max(a, b) <= min(a, b) + ScientificNotation(Float(1, [0]), 0):
            # now we need the difference between a and b > 1 to avoid the negative exponent
            return max(a, b) + ScientificNotation(Float(1, [0]), 0), min(a, b)
        return max(a, b), min(a, b)
    
    def _preprocess_compare(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_same_type(input_numbers, "compare")
        if isinstance(a, Fraction) and isinstance(b, Fraction):
            if a.num > a.den:
                a = type(a)(a.den, a.num)
            if b.num > b.den:
                b = type(b)(b.den, b.num)
        return a, b
    
    def _preprocess_max(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        return self._preprocess_compare(input_numbers)
    
    def _preprocess_max_hard(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        return self._preprocess_max(input_numbers)
    
    def _preprocess_min(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        return self._preprocess_compare(input_numbers)
    
    def _preprocess_min_hard(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        return self._preprocess_min(input_numbers)
    
    def _preprocess_multiply_hard(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        return self._preprocess_multiply(input_numbers)
    
    def _preprocess_multiply_easy(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_larger_digit_first(input_numbers, "Multiply")
        b = b.trunc_digit(MULTIPLY_EASY_MAX_DIGIT)
        return a, b
    
    def _preprocess_multiply(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        return self._preprocess_same_type(input_numbers, "Multiply")
    
    def _preprocess_digit_max(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        return self._preprocess_same_type(input_numbers, "Digit_max")
    
    def _preprocess_digit_min(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        return self._preprocess_same_type(input_numbers, "Digit_min")
    
    def _preprocess_digit_add(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        return self._preprocess_same_type(input_numbers, "Digit_add")
    
    def _preprocess_get_digit(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, int]:
        a, b = self._preprocess_larger_digit_first(input_numbers, "Get_digit")
        digit = self._tmp_random_rng(seed=int(b.part_str()[0])).randint(0, sum([len(s) for s in a.part_str()])-1)
        return a, digit
    
    def _preprocess_length(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, None]:
        a, b = self._preprocess_larger_digit_first(input_numbers, "Length")
        return a, None
    
    def _den_to_nonzero(self, den: T) -> T:
        if den == Integer(0):
            assert isinstance(den, Integer)
            return type(den)(1) # type: ignore
        elif den == Float(0, [0]):
            assert isinstance(den, Float)
            return type(den)(1, [0]) # type: ignore
        elif den == Fraction(0, 1):
            assert isinstance(den, Fraction)
            return type(den)(1, 1) # type: ignore
        elif den == ScientificNotation(Float(0, [0]), 0):
            assert isinstance(den, ScientificNotation)
            return type(den)(Float(1, [0]), 0) # type: ignore
        else:
            return den
    
    def _preprocess_truediv(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_same_type(input_numbers, "Truediv")
        return a, self._den_to_nonzero(b)
    
    def _preprocess_floordiv(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_larger_first(input_numbers, "Floordiv")
        return a, self._den_to_nonzero(b)
    
    def _preprocess_mod(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_larger_first(input_numbers, "Mod")
        # Here, we need to make sure the b is not too closed to a to avoid the mod task can be solved by easily subtracting a and b.
        assert isinstance(a, Integer) and isinstance(b, Integer), "Mod task only support two Integer inputs."
        if b.digit > (upper_bound:=(int(0.7 * a.digit) + 1)):
            b = b.trunc_digit(upper_bound)
        return a, self._den_to_nonzero(b)
    
    def _preprocess_mod_easy(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, NumberBasic]:
        a, b = self._preprocess_larger_first(input_numbers, "Mod")
        assert isinstance(a, Integer) and isinstance(b, Integer), "Mod_easy task only support two Integer inputs."
        if b.digit > MOD_EASY_MAX_DIGIT:
            b = b.trunc_digit(MOD_EASY_MAX_DIGIT)
        return a, self._den_to_nonzero(b)
    
    def _preprocess_to_float(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, None]:
        a, b = self._preprocess_larger_digit_first(input_numbers, "To_float")
        if isinstance(a, Fraction) and isinstance(b, Fraction):
            exp_2 = self._tmp_random_rng(seed=int(b.part_str()[0])).randint(0, max(a.digit, b.digit))
            exp_5 = self._tmp_random_rng(seed=int(b.part_str()[1])).randint(0, max(a.digit, b.digit))
            num = max(a.num, a.den)
            return Fraction(num, 2**exp_2 * 5**exp_5), None
        elif isinstance(a, ScientificNotation) and isinstance(b, ScientificNotation):
            # make the exponent shorter to avoid too long float
            return ScientificNotation(a.mantissa, self._tmp_random_rng(seed=int(b.part_str()[0])).randint(0, min(a.exponent, a.digit))), None
        else:
            return a, None
        
    def _preprocess_to_scient(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, None]:
        a, b = self._preprocess_larger_digit_first(input_numbers, "To_scient")
        if isinstance(a, Float) and a < Float(1, [0]):
            a = a + Float(1, [0])
        max_zero = int(0.5 * a.digit)
        zero_ = self._tmp_random_rng(seed=int(b.part_str()[0])).randint(0, max_zero)
        if isinstance(a, Integer):
            # make there are more zero
            a_value = a.value - a.value % 10**zero_
            a = Integer(a_value)
        elif isinstance(a, Float):
            if a.int_part == 0:
                a = Float(1, a.dec_part) # avoid negative exponent
        else:
            raise TypeError(f"Type {type(a)} is not supported to convert to ScientificNotation.")
        return a, None
    
    def _preprocess_count(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, int]:
        a, b = self._preprocess_larger_digit_first(input_numbers, "count")
        seed = int(b.part_str()[0])
        count = self._tmp_random_rng(seed=seed).randint(0, 9)
        return a, count
    
    def _preprocess_sig(self, input_numbers: Sequence[NumberBasic | int]) -> tuple[NumberBasic, int]:
        a, b = self._preprocess_larger_digit_first(input_numbers, "sig")
        seed = int(b.part_str()[0])
        if isinstance(a, Integer):
            max_significant = a.digit
        elif isinstance(a, Float):
            if a == Float(0, [0]):
                max_significant = 2
            else:
                max_significant = len(str(a.to_int_rep()[0]).strip('0'))
        else: 
            raise TypeError(f"The input number of task sig. fig. should be either Integer or Float, but get {type(a)}.")
        # max_significant = a.digit if isinstance(a, Integer) else sum([len(p) for p in a.part_str()])
        sig = self._tmp_random_rng(seed=seed).randint(2, max(2, max_significant))
        if isinstance(a, Float) and a.int_part == 0:
            a = Float(1, a.dec_part)
        return a, sig

    def _tmp_random_rng(self, seed:int):
        return random.Random(seed)

    def calc_result(self, a: NumberBasic, b: NumberBasic | int | None) -> NumberBasic | int:
        assert type(a).__name__ == self.input_a_type, f"First input type {type(a).__name__} inconsistent with task define {self.input_a_type}"
        if self.input_b_type == "none":
            assert b is None, f"Second input must be None if task define the 'input_b_type' as 'none'"
        else:
            assert type(b).__name__ == self.input_b_type, f"Second input type {type(b).__name__} inconsistent with task define {self.input_b_type}"
        result = self._calc_result(a, b)
        assert type(result).__name__ == self.output_type, f"Output type {type(result).__name__} inconsistent with task define {self.output_type}. Check the calculation logic."
        return result
    
    def instruction(self) -> str:
        instruction_dict = {
            "add": "Add two numbers: ",
            "add_easy": "Add two numbers: ",
            "sub": "Subtract two numbers: ",
            "max": "Get the maximal number: ",
            "max_hard": "Get the maximal number: ",
            "min": "Get the mininal number: ",
            "min_hard": "Get the mininal number: ",
            "multiply_easy": "Multiply two numbers: ",
            "multiply_hard": "Multiply two numbers: ",
            "digit_max": "Compare two numbers digit by digit and return the larger digit at each position, treating any missing digits as 0. ",
            "digit_min": "Compare two numbers digit by digit and return the smaller digit at each position, treating any missing digits as 0. ",
            "digit_add": "The task is to add two given numbers digit by digit and return the result modulo 10 (ignoring carry), treating any missing digits as 0. ",
            "get_digit": "Get the digit at the given position (from left to right, starting from 0). ",
            "length": "The total number of digits of ",
            "truediv": "Divide two numbers and return the result as a fraction. ",
            "floordiv": "Divide two numbers and return the result as an integer. ",
            "mod": "Divide two numbers and return the remainder. ",
            "mod_easy": "Divide two numbers and return the remainder. ",
            "to_float": "Convert the number to float: ",
            "to_scient": "Convert the number to scientific notation: ",
            "count": "Count the number of the given digit in the given number: ",
            "sig": "Convert the number to scientific notation: ",
        }
        if self.input_a_type == "float":
            instruction_dict["digit_max"] = "Compare two numbers digit by digit (aligning the decimal points) and return the larger digit at each position, treating any missing digits as 0. "
            instruction_dict["digit_min"] = "Compare two numbers digit by digit (aligning the decimal points) and return the smaller digit at each position, treating any missing digits as 0. "
            instruction_dict["digit_add"] = "The task is to add two given numbers digit by digit (aligning the decimal points) and return the result modulo 10 (ignoring carry), treating any missing digits as 0. "
            instruction_dict["get_digit"] = "Get the digit at the given position (from left to right, starting from 0, ignoring the decimal point)"
        
        type_hint = {
            "int": "Directly return the answer as an integer without any comma separator, like 123 . ",
            "Integer": "Directly return the answer as an integer without any comma separator, like 123 . ",
            "Float": "Directly return the answer as a float without any comma separator, like 10.4 . ",
            "Fraction": "Directly return the answer as an **irreducible** fraction without any comma separator, like 7/13 . ",
            "ScientificNotation": "Directly return the answer as a scientific notation without any comma separator, like 1.23e4 . The float part should be in the range [1, 10). ",
        }
        
        return type_hint[self.output_type] + instruction_dict[self.task]
        
    def operator(self) -> str:
        operator_dict = {
            "add": "+",
            "add_easy": "+",
            "sub": "-",
            "max": "and",
            "max_hard": "and",
            "min": "and",
            "min_hard": "and",
            "multiply_easy": "*",
            "multiply_hard": "*",
            "digit_max": "and",
            "digit_min": "and",
            "digit_add": "digit add",
            "get_digit": "at position",
            "truediv": "/",
            "floordiv": "//",
            "mod": "%",
            "mod_easy": "%",
            "count": "count the occurrence time of digit",
            "sig": "and keep significant figures as"
        }
        return " " + operator_dict[self.task] + " "
    
    def answer(self) -> str:
        return " = "

    def _calc_result(self, a: NumberBasic, b: NumberBasic | int | None) -> NumberBasic:
        assert hasattr(self, f"_calc_{self.task}"), f"Method _calc_{self.task} not found."
        return getattr(self, f"_calc_{self.task}")(a, b)

    def _calc_add(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return a + b
    
    def _calc_add_easy(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return self._calc_add(a, b)
    
    def _calc_sub(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return a - b

    def _check_compare(self, a: NumberBasic, b: NumberBasic) -> None:
        if not hasattr(a, "__gt__") or not hasattr(a, "__lt__"):
            raise AttributeError(
                f"The type {type(a)} does not support '>' and '<'."
            )

    def _calc_max(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        self._check_compare(a, b)
        return a if (a > b) else b
    
    def _calc_max_hard(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return self._calc_max(a, b)

    def _calc_min(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        self._check_compare(a, b)
        return a if (a < b) else b
    
    def _calc_min_hard(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return self._calc_min(a, b)

    def _calc_multiply(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return a * b
    
    def _calc_multiply_easy(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return self._calc_multiply(a, b)
    
    def _calc_multiply_hard(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return self._calc_multiply(a, b)

    def _calc_digit_operator(self, a: NumberBasic, b: NumberBasic, operator: Callable[[int, int], int]) -> NumberBasic:
        assert type(a) == type(b), f"Type of a {type(a)} and b {type(b)} must be the same."
        strs_a, strs_b = a.part_str(), b.part_str()
        directs = a.default_read_direct()
        str_answer = []
        for str_a, str_b, direct in zip(strs_a, strs_b, directs):
            if direct == "right":
                str_answer.append(''.join(([str(operator(int(a_digit), int(b_digit))) for a_digit, b_digit in itertools.zip_longest(str_a[::-1], str_b[::-1], fillvalue='0')][::-1])))
            elif direct == "left":
                str_answer.append(''.join(([str(operator(int(a_digit), int(b_digit))) for a_digit, b_digit in itertools.zip_longest(str_a, str_b, fillvalue='0')])))
            else:
                raise ValueError(f'Invalid direct {direct}.')
        return a.from_string(
            "".join([
                (
                    (str_part + a.sep_str()[i]) # type: ignore # If sep_str() is None, len(str_answer) == 1, so this is safe.
                    if i != (len(str_answer) - 1)
                    else str_part
                )
                for i, str_part in enumerate(str_answer)
            ])
        )

    def _calc_digit_max(self, a: NumberBasic, b: NumberBasic):
        return self._calc_digit_operator(a, b, max)
    
    def _calc_digit_min(self, a: NumberBasic, b: NumberBasic):
        return self._calc_digit_operator(a, b, min)
    
    def _calc_digit_add(self, a: NumberBasic, b: NumberBasic):
        return self._calc_digit_operator(a, b, (lambda x, y: (x + y) % 10))
    
    def _calc_get_digit(self, a: NumberBasic, b: int) -> int:
        return int("".join(a.part_str())[b])
    
    def _calc_length(self, a: NumberBasic, b: None) -> int:
        return len("".join(a.part_str())) 
    
    def _calc_floordiv(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return a // b # type: ignore
    
    def _calc_truediv(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return a / b # type: ignore
    
    def _calc_mod(self, a: NumberBasic, b: NumberBasic) -> NumberBasic:
        return a % b # type: ignore
    
    def _calc_mod_easy(self, a:NumberBasic, b: NumberBasic) -> NumberBasic:
        return self._calc_mod(a, b)
    
    def _calc_count(self, a: NumberBasic, b: int) -> int:
        return sum([int(digit) == b for digit in "".join(a.part_str())])
    
    def _calc_to_float(self, a: NumberBasic, b: None) -> Float:
        if isinstance(a, (Fraction, ScientificNotation)):
            return a.to_float()
        else:
            raise TypeError(f"Type {type(a)} is not supported to convert to Float.")
        
    def _calc_to_scient(self, a: NumberBasic, b: None) -> ScientificNotation:
        if isinstance(a, Integer):
            a = Float(a.value, [0])
        if isinstance(a, Float):
            return a.to_scient()
        else:
            raise TypeError(f"Type {type(a)} is not supported to convert to ScientificNotation.")
        
    def _calc_sig(self, a: NumberBasic, b: int) -> ScientificNotation:
        if b <= 1:
            raise ValueError(f"Now we do not support sig. fig with less than 2.")
        if isinstance(a, Integer):
            a = Float(a.value, [0])
        if isinstance(a, Float):
            return a.to_scient(sig=b)
        else:
            raise TypeError(f"Type {type(a)} is not supported to convert to ScientificNotation.")