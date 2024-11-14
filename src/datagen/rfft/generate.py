from datagen.rfft.tasks import Dataset_Generator
import json, argparse


parser = argparse.ArgumentParser("""Generate rfft datasets.""")
parser.add_argument("--model_path", type=str, help="The path of the model will be used to tokenize the data.")

args = parser.parse_args()

task_names = {
    'add_Integer_Integer_Integer': 20,
    'add_Float_Float_Float': 8,
    'add_Fraction_Fraction_Fraction': 4,
    'add_easy_Fraction_Fraction_Fraction': 4,
    'add_ScientificNotation_ScientificNotation_ScientificNotation': 3,
    'sub_Integer_Integer_Integer': 20,
    'sub_Float_Float_Float': 8,
    'sub_Fraction_Fraction_Fraction': 6,
    'sub_ScientificNotation_ScientificNotation_ScientificNotation': 6,
    'max_Integer_Integer_Integer': 100,
    'max_Float_Float_Float': 60,
    'max_Fraction_Fraction_Fraction': 5,
    'max_ScientificNotation_ScientificNotation_ScientificNotation': 40,
    'max_hard_Integer_Integer_Integer': 30,
    'max_hard_Float_Float_Float': 50,
    'max_hard_ScientificNotation_ScientificNotation_ScientificNotation': 20,
    'multiply_hard_Integer_Integer_Integer': 15,
    'multiply_hard_Float_Float_Float': 4,
    'multiply_hard_Fraction_Fraction_Fraction': 3,
    'multiply_hard_ScientificNotation_ScientificNotation_ScientificNotation': 4,
    'multiply_easy_Integer_Integer_Integer': 15,
    'multiply_easy_Float_Float_Float': 4,
    'multiply_easy_Fraction_Fraction_Fraction': 3,
    'multiply_easy_ScientificNotation_ScientificNotation_ScientificNotation': 4,
    'digit_max_Integer_Integer_Integer': 20,
    'get_digit_Integer_int_int': 40,
    'get_digit_Float_int_int': 20,
    'length_Integer_none_int': 50,
    'floordiv_Integer_Integer_Integer': 6,
    'mod_Integer_Integer_Integer': 6,
    'mod_easy_Integer_Integer_Integer': 6
}

task_names = {
    'max_Integer_Integer_Integer': 100,
    'max_hard_Integer_Integer_Integer': 30,
}

model_path = args.model_path
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("Tokenizer loaded")


def transform(stage):
    rfft = {}
    with open(f"benchmark/tasks/{stage}.json", "r") as f:
        data = json.load(f)
    cnt = 0
    
    for task_name, data2 in data.items():
        if task_name not in task_names.keys():
            continue
        print(task_name)
        t1 = {}
        Generator = Dataset_Generator(task_name)
        for digit, tests in data2.items():
            if int(digit) > task_names[task_name]:
                continue
            cnt = 0
            temp = []
            
            for test in tests:
                # if (cnt % 5000 == 0):
                #     print(cnt)
                if cnt > 15000:
                    break
                sample = Generator.rfft_IO(test)
                final_str = '='.join(sample["input"].split('=')[:-1]) + '\n\n## Response:\n' + sample["output"]

                tokens = tokenizer.tokenize(final_str)
                token_count = len(tokens)
                if token_count > 2000:
                    continue
                cnt+=1
                temp.append(final_str)
            
            t1[digit] = temp
        rfft[task_name] = t1
        
    with open(f"benchmark/tasks/rfft_{stage}.json", "w") as f:
        json.dump(rfft, f)

transform("train")
transform("valid")
transform("test")
