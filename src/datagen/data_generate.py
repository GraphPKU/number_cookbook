import random
from tqdm.autonotebook import tqdm
from number_model.tokenizer import NumberTokenizer
from collections.abc import Sequence
import typing
from typing import Literal, overload, Type
import os
from task import Task

from numbers_class import NumberBasic, Domain, ScientificNotation, Integer, Float
DataType = list[list[NumberBasic]] # Outer list is the samples, inner list is the numbers in the sample

PROB_DIFF_FLOAT_IN_INT = 0.3
PROB_DIFF_SCIENTIFIC_IN_EXP = 0.2

class DataGenerator:
    def __init__(self, domain: Domain, nums: int | Sequence[int], min_len: int, max_len: int, same_len: bool | None = None, random_seed: int | None = None, valid_lens: Sequence[int] | None = None, valid_nums: int | Sequence[int] | None = None, test_lens: Sequence[int] | None = None, test_nums: int | Sequence[int] | None = None, test_shorter_len: int | float | None = None, harder_compare: bool = False):
        self.domain: Domain = domain
        self.min_len = min_len
        self.max_len = max_len
        self.same_len = same_len # If None, with both same length and shorter; If True, only with same length; If False, without same length
        self.valid_lens = valid_lens
        self.valid_nums = valid_nums
        self.test_lens = test_lens
        self.test_nums = test_nums
        self.test_shorter_len = test_shorter_len
        self.harder_compare = harder_compare
        
        if isinstance(nums, int):
            self.nums: Sequence[int] = [nums for i in range(min_len, max_len+1)]
        elif isinstance(nums, Sequence):
            if len(nums) != (max_len - min_len + 1):
                raise ValueError("The length of nums should be equal to max_len - min_len + 1.")
            self.nums = nums
        else:
            raise TypeError("The nums should be an int or a list of int.")
        
        self.add_valid_and_test_in_nums()
        
        self.set_random_seed(random_seed)
        self.data: None | DataType = None
        self.valid_data: None | DataType = None
        self.test_data: None | DataType = None

    def set_random_seed(self, random_seed: int | None = None) -> None:
        self.random_rng = random.Random(random_seed)

    def add_valid_and_test_in_nums(self) -> None:
        if self.valid_lens is not None:
            valid_lens = self.valid_lens
            assert self.valid_nums is not None, "valid_nums is required when giving a valid_lens"
            if isinstance(self.valid_nums, int):
                valid_nums = [self.valid_nums for i in range(len(self.valid_lens))]
            elif isinstance(self.valid_nums, Sequence):
                if len(self.valid_nums) != len(self.valid_lens):
                    raise ValueError("The length of valid_nums should be equal to the length of valid_lens.")
                valid_nums = self.valid_nums
            else:
                raise TypeError("The valid_nums should be an int or a list of int.")
        else:
            valid_lens = []
            valid_nums = []
            
        if self.test_lens is not None:
            test_lens = self.test_lens
            assert self.test_nums is not None, "test_nums is required when giving a valid_lens"
            if isinstance(self.test_nums, int):
                test_nums = [self.test_nums for i in range(len(self.test_lens))]
            elif isinstance(self.test_nums, Sequence):
                if len(self.test_nums) != len(self.test_lens):
                    raise ValueError("The length of test_nums should be equal to the length of test_lens.")
                test_nums = self.test_nums
            else:
                raise TypeError("The test_nums should be an int or a list of int.")
        else:
            test_lens = []
            test_nums = []
            
        full_min_len = typing.cast(int, min(self.min_len, min(valid_lens) if valid_lens else float("inf"), min(test_lens) if test_lens else float("inf")))
        full_max_len = typing.cast(int, max(self.max_len, max(valid_lens) if valid_lens else -float("inf"), min(test_lens) if test_lens else -float("inf")))
        full_nums: Sequence[int] = []
        for i in range(full_min_len, full_max_len+1):
            num = 0
            if i >= self.min_len and i <= self.max_len:
                num += self.nums[i-self.min_len]
            if i in valid_lens:
                num += valid_nums[valid_lens.index(i)]
            if i in test_lens:
                num += test_nums[test_lens.index(i)]
            full_nums.append(num)
                
        self.nums: Sequence[int] = full_nums
        self.min_len = full_min_len
        self.max_len = full_max_len
        
    def unique(self, data: DataType) -> tuple[DataType, list[int]]:
        # convert list to tuple to make it hashable
        tupled_data = [tuple(d) for d in data]
        unique_dict = {}
        for idx, item in enumerate(tupled_data):
            if item not in unique_dict:
                unique_dict[item] = idx
        index = list(unique_dict.values())
        index.sort()
        unique_data: DataType = [data[i] for i in index]
        return unique_data, index
    
    def make_harder_compare(self, pair: list[NumberBasic]) -> list[NumberBasic]:
        if not self.harder_compare:
            return pair
        a, b = pair[0], pair[1]
        if isinstance(a, Integer) and isinstance(b, Integer):
            assert self.same_len, "Harder compare between Integers needs same length."
            assert a.digit == b.digit
            diff_digit = self.random_rng.randint(0, a.digit - 1)
            b = type(b).from_string(str(a)[:diff_digit] + str(b)[diff_digit: ])
            return [a, b]
        elif isinstance(a, Float) and isinstance(b, Float):
            if self.random_rng.random() < PROB_DIFF_FLOAT_IN_INT:
                return [a, b]
            else:
                diff_digit = self.random_rng.randint(0, min(a.digit, b.digit) - 1)
                b_dec = a.dec_part[:diff_digit] + b.dec_part[diff_digit:]
                b = type(b)(a.int_part, b_dec) # make a and b share the same int part and some beginning decimal part
                return [a, b]
        elif isinstance(a, ScientificNotation) and isinstance(b, ScientificNotation):
            if self.random_rng.random() > PROB_DIFF_SCIENTIFIC_IN_EXP:
                b_exp = a.exponent
            else:
                b_exp = b.exponent
            diff_digit = self.random_rng.randint(-1, min(a.digit, b.digit) - 1)
            if diff_digit == -1:
                b = type(b)(b.mantissa, b_exp)
            else:
                b_man = Float(a.mantissa.int_part, a.mantissa.dec_part[:diff_digit] + b.mantissa.dec_part[diff_digit:])
                b = type(b)(b_man, b_exp)
            return [a, b]
        else:
            raise TypeError(f"Invalid type a {type(a)} and type b {type(b)} for harder compare.")
        
    
    def generate(self) -> tuple[DataType, DataType | None, DataType | None]:
        generated_data = []
        valid_data = []
        test_data = []
        for (num, lens) in tqdm(zip(self.nums, range(self.min_len, self.max_len+1))):
            if num == 0:
                continue
            class_ = NumberBasic.get_subclass(domain=self.domain)
            a = [class_.random_generate(lens, random_rng=self.random_rng) for i in range(num)]
            
            shorter_lower_bound = self.min_len
            
            if self.valid_lens is None or lens not in self.valid_lens:
                valid_num = 0
            elif isinstance(self.valid_nums, int):
                valid_num = self.valid_nums
            else:
                valid_num = self.valid_nums[self.valid_lens.index(lens)]
                
            if self.test_lens is None or lens not in self.test_lens:
                test_num = 0
            elif isinstance(self.test_nums, int):
                test_num = self.test_nums
            else:
                test_num = self.test_nums[self.test_lens.index(lens)]
                
            if num == valid_num + test_num and self.test_shorter_len is not None:
                test_shorter_len = self.test_shorter_len if isinstance(self.test_shorter_len, int) else int(self.test_shorter_len * lens)
                shorter_lower_bound = test_shorter_len
            
            if self.same_len:
                b_lens = [lens for _ in range(num)]
            elif self.same_len is not None: # same_len is False
                if lens == shorter_lower_bound:
                    continue
                b_lens = [self.random_rng.randint(shorter_lower_bound, lens - 1) for _ in range(num)]
            else:
                # b_lens is the random length between min_len and the current lens
                b_lens = [self.random_rng.randint(shorter_lower_bound, lens) for _ in range(num)]
            b = [class_.random_generate(bl, random_rng=self.random_rng) for bl in b_lens]
            data: DataType = [self.make_harder_compare([a_ele, b_ele]) for a_ele, b_ele in zip(a,b)] # (num, 2)
            # # first, make sure a >= b to remove duplicated pairs
            # data[data[:,0] < data[:,1]] = data[data[:,0] < data[:,1]][:,::-1]
            # then remove duplicated pairs
            data, indice = self.unique(data)
            b_lens = [b_lens[i] for i in indice]
            
            # If self.test_shorter_len is not None, we need to count the elements in b_lens >= test_shorter_len.
            
            if self.test_shorter_len is not None:
                test_shorter_len = self.test_shorter_len if isinstance(self.test_shorter_len, int) else int(self.test_shorter_len * lens)
                index_satisfy_test_shorter_len = [ii for ii, bl in enumerate(b_lens) if bl >= test_shorter_len]
                if len(index_satisfy_test_shorter_len) < valid_num + test_num:
                    print(f"Warning! The number of pairs with b_lens >= test_shorter_len ({len(index_satisfy_test_shorter_len)}) is less than valid_num + test_num ({valid_num + test_num}).")
                test_num = min(test_num, len(index_satisfy_test_shorter_len) // 2)
                valid_num = min(valid_num, len(index_satisfy_test_shorter_len) // 2)
            else:
                index_satisfy_test_shorter_len = list(range(len(data)))
            
            # reverse some pairs
            reverse = self.random_rng.choices([True, False], k=len(data))
            reverse = [(reverse[i] and (b_lens[i] != lens)) for i in range(len(reverse))] # only reverse the pair with different length
            # reverse a[i] and b[i] if reverse[i] is True
            for i in range(len(reverse)):
                if reverse[i]:
                    data[i] = data[i][::-1]
            
            # some sample should be in valid or test set
            valid_index = self.random_rng.sample(index_satisfy_test_shorter_len, k=valid_num)
            valid_data.append([data[i] for i in valid_index])
            index_satisfy_test_shorter_len = [jj for jj in index_satisfy_test_shorter_len if jj not in valid_index]
            
            test_index = self.random_rng.sample(index_satisfy_test_shorter_len, k=test_num)
            test_data.append([data[i] for i in test_index])
            
            data = [data[i] for i in range(len(data)) if (i not in test_index and i not in valid_index)]
            
            generated_data.append(data)
            tqdm.write(f"Length: {lens}, training num: {len(data)}, valid num: {valid_num}, test num: {test_num}.")
        self.data = sum(generated_data, start=[]) # (nums, 2)
        self.valid_data = sum(valid_data, start=[]) if valid_data else None
        self.test_data = sum(test_data, start=[]) if test_data else None
        return self.data, self.valid_data, self.test_data
    
    @overload
    def check_all_token_observe(self, tokenizer: NumberTokenizer, return_unobserved: Literal[False] = False) -> bool:
        ...
        
    @overload
    def check_all_token_observe(self, tokenizer: NumberTokenizer, return_unobserved: Literal[True]) -> tuple[bool, set]:
        ...
    
    def check_all_token_observe(self, tokenizer: NumberTokenizer, return_unobserved: bool = False) -> tuple[bool, set] | bool:
        # check whether all tokens can be trained
        assert self.data is not None, "Please generate first."
        observed = set(list(range(tokenizer.num_addition_token, tokenizer.num_token)))
        print("Checking whether all tokens can be trained ...")
        for sample in tqdm(self.data):
            for number in sample:
                observed -= set(tokenizer.encode_number(number))
                if not observed:
                    break
        if return_unobserved:
            return not observed, observed
        else:
            return not observed
    
    def save_dataset(self, path: str, save_as_txt: bool = False) -> None:
        os.makedirs(os.path.join(path), exist_ok=True)
            
        assert self.data is not None, "Please generate first."
        if not save_as_txt:
            import pickle
            with open(os.path.join(path, "train.pkl") ,"wb") as f:
                pickle.dump(self.data, f)
            if self.valid_data is not None:
                with open(os.path.join(path, "valid.pkl") ,"wb") as f:
                    pickle.dump(self.valid_data, f)
            if self.test_data is not None:
                with open(os.path.join(path, "test.pkl") ,"wb") as f:
                    pickle.dump(self.test_data, f)
        else:
            with open(os.path.join(path, "train.txt"), "w") as f:
                for sample in self.data:
                    f.write(sample[0].to_string() + "\t" + sample[1].to_string() + "\n")
            if self.valid_data is not None:
                with open(os.path.join(path, "valid.txt"), "w") as f:
                    for sample in self.valid_data:
                        f.write(sample[0].to_string() + "\t" + sample[1].to_string() + "\n")
            if self.test_data is not None:
                with open(os.path.join(path, "test.txt"), "w") as f:
                    for sample in self.test_data:
                        f.write(sample[0].to_string() + "\t" + sample[1].to_string() + "\n")
                        
        print(f"Save generated dataset in {os.path.abspath(path)}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser("""Generate number dataset. A binary file will be saved in the path where a list of list of NumberBasic is stored. The """)
    parser.add_argument('-d', '--domain', type=str, help='The representation of numbers.', choices=['Integer', 'Float', 'Fraction', 'ScientificNotation'])
    parser.add_argument('-n', '--nums', nargs='+', type=int, help='The number of training samples. Int or a int list. If a list, the length should be equal to max_len - min_len + 1.')
    parser.add_argument('--min_len', type=int, required=True, help='The minimal length of training numbers.')
    parser.add_argument('--max_len', type=int, required=True, help='The maximal length of training numbers.')
    parser.add_argument('--same_len', type=str, default=None, help='Whether generate number pairs with the same length. If True, always same; If False, always different length. Default: None, with both same and different length.')
    parser.add_argument('--random_seed', type=int, default=20240627)
    parser.add_argument('--min_valid_len', type=int, required=True)
    parser.add_argument('--max_valid_len', type=int, required=True)
    parser.add_argument('--valid_nums', nargs='+', type=int, help="Int of a list of int.")
    parser.add_argument('--min_test_len', type=int, required=True)
    parser.add_argument('--max_test_len', type=int, required=True)
    parser.add_argument('--test_nums', nargs='+', type=int, help="Int of a list of int.")
    parser.add_argument('--test_shorter_len', type=eval, help="The length (if int) or ratio (if float) of the length of the shorter number in the test set. If None, no such constraint.")
    parser.add_argument('--save_path', type=str, default='data')
    parser.add_argument('--skip_check_token', action='store_true', help='Whether to skip the process to check which tokens are not in the training set.')
    parser.add_argument('--harder_compare', action='store_true', help='Whether to generate harder compare task.')
    parser.add_argument('--save_as_txt', action='store_true', help='Whether to save the dataset as txt file.')
    return parser.parse_args()
        
if __name__ == "__main__":
    args = parse_args()
    # Some preprocess for args
    nums = args.nums[0] if len(args.nums) == 1 else args.nums
    valid_nums = args.valid_nums[0] if len(args.valid_nums) == 1 else args.valid_nums
    test_nums = args.test_nums[0] if len(args.test_nums) == 1 else args.test_nums
    if args.same_len is None:
        same_len = None
    else:
        same_len = True if args.same_len.lower() in ['y', 'yes', 't', 'true', 'on', '1'] else False

    data_generator = DataGenerator(
            domain=args.domain, 
            nums=nums, 
            min_len=args.min_len, 
            max_len=args.max_len, 
            same_len=same_len, 
            random_seed=args.random_seed, 
            valid_lens=list(range(args.min_valid_len, args.max_valid_len+1)), 
            valid_nums=valid_nums, 
            test_lens=list(range(args.min_test_len, args.max_test_len+1)), 
            test_nums=test_nums,
            test_shorter_len=args.test_shorter_len,
            harder_compare=args.harder_compare,
            )
    data_generator.generate()
    task = Task("add", args.domain, args.domain, args.domain)
    tokenizer = NumberTokenizer(task=task, digit=3)
    if not args.skip_check_token:
        full, unobserved = data_generator.check_all_token_observe(tokenizer, return_unobserved=True)
        if unobserved:
            print(f"Some tokens {list(map(tokenizer.decode_one_token, list(unobserved)))} are not included in the training set.")
    data_generator.save_dataset(args.save_path, args.save_as_txt)
