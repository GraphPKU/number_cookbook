import os
from task import Task

benchmark_path = "benchmark"
dataset_path = os.path.join(benchmark_path, "numbers")
random_seed = 208171

dataset_list = [
    ["-d", "Integer", "-n", "100000", "--min_len", "1", "--max_len", "8", "--min_valid_len", "3", "--max_valid_len", "20", "--valid_nums", "1000", "--min_test_len", "3", "--max_test_len", "20", "--test_nums", "1000", "--test_shorter_len", "0.5", "--save_path", os.path.join(dataset_path, "Integer", "add"), "--skip_check_token", '--random_seed', str(random_seed+0)],
    ["-d", "Float", "-n", "100000", "--min_len", "1", "--max_len", "8", "--min_valid_len", "3", "--max_valid_len", "20", "--valid_nums", "1000", "--min_test_len", "3", "--max_test_len", "20", "--test_nums", "1000", "--test_shorter_len", "0.5", "--save_path", os.path.join(dataset_path, "Float", "add"), "--skip_check_token", '--random_seed', str(random_seed+1)],
    ["-d", "Fraction", "-n", "100000", "--min_len", "1", "--max_len", "8", "--min_valid_len", "1", "--max_valid_len", "20", "--valid_nums", "1000", "--min_test_len", "1", "--max_test_len", "20", "--test_nums", "1000", "--test_shorter_len", "0.5", "--save_path", os.path.join(dataset_path, "Fraction", "add"), "--skip_check_token", '--random_seed', str(random_seed+2)],
    ["-d", "ScientificNotation", "-n", "100000", "--min_len", "1", "--max_len", "8", "--min_valid_len", "3", "--max_valid_len", "20", "--valid_nums", "1000", "--min_test_len", "3", "--max_test_len", "20", "--test_nums", "1000", "--test_shorter_len", "0.5", "--save_path", os.path.join(dataset_path, "ScientificNotation", "add"), "--skip_check_token", '--random_seed', str(random_seed+3)],

    # for some domain, the compare is much more easier than add, so we set a longer length for the test set.
    ["-d", "Integer", "-n", "100000", "--min_len", "1", "--max_len", "20", "--min_valid_len", "5", "--max_valid_len", "100", "--valid_nums", "1000", "--min_test_len", "5", "--max_test_len", "100", "--test_nums", "1000", "--test_shorter_len", "0.5", "--save_path", os.path.join(dataset_path, "Integer", "compare"), "--skip_check_token", '--random_seed', str(random_seed+4)],
    ["-d", "Float", "-n", "100000", "--min_len", "1", "--max_len", "20", "--min_valid_len", "3", "--max_valid_len", "100", "--valid_nums", "1000", "--min_test_len", "3", "--max_test_len", "100", "--test_nums", "1000", "--test_shorter_len", "0.5", "--save_path", os.path.join(dataset_path, "Float", "compare"), "--skip_check_token", '--random_seed', str(random_seed+5)],
    ["-d", "ScientificNotation", "-n", "100000", "--min_len", "1", "--max_len", "20", "--min_valid_len", "3", "--max_valid_len", "100", "--valid_nums", "1000", "--min_test_len", "3", "--max_test_len", "100", "--test_nums", "1000", "--test_shorter_len", "0.5", "--save_path", os.path.join(dataset_path, "ScientificNotation", "compare"), "--skip_check_token", '--random_seed', str(random_seed+6)],

    # A harder version compare, where numbers share more same digit
    ["-d", "Integer", "-n", "100000", "--min_len", "1", "--max_len", "20", "--min_valid_len", "5", "--max_valid_len", "100", "--valid_nums", "1000", "--min_test_len", "5", "--max_test_len", "100", "--test_nums", "1000", "--test_shorter_len", "0.5", "--save_path", os.path.join(dataset_path, "Integer", "compare_harder"), "--skip_check_token", "--harder_compare", "--same_len", 'true', '--random_seed', str(random_seed+7)],
    ["-d", "Float", "-n", "100000", "--min_len", "1", "--max_len", "20", "--min_valid_len", "3", "--max_valid_len", "100", "--valid_nums", "1000", "--min_test_len", "3", "--max_test_len", "100", "--test_nums", "1000", "--test_shorter_len", "0.5", "--save_path", os.path.join(dataset_path, "Float", "compare_harder"), "--skip_check_token", "--harder_compare", '--random_seed', str(random_seed+8)],
    ["-d", "ScientificNotation", "-n", "100000", "--min_len", "1", "--max_len", "20", "--min_valid_len", "3", "--max_valid_len", "100", "--valid_nums", "1000", "--min_test_len", "3", "--max_test_len", "100", "--test_nums", "1000", "--test_shorter_len", "0.5", "--save_path", os.path.join(dataset_path, "ScientificNotation", "compare_harder"), "--skip_check_token", "--harder_compare", '--random_seed', str(random_seed+9)],
]

task_dataset_list = [
    (Task("add", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "add")),
    (Task("add", "Float", "Float", "Float"), os.path.join(dataset_path, "Float", "add")),
    (Task("add", "Fraction", "Fraction", "Fraction"), os.path.join(dataset_path, "Fraction", "add")),
    (Task("add_easy", "Fraction", "Fraction", "Fraction"), os.path.join(dataset_path, "Fraction", "add")),
    (Task("add", "ScientificNotation", "ScientificNotation", "ScientificNotation"), os.path.join(dataset_path, "ScientificNotation", "add")),
    
    (Task("sub", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "add")),
    (Task("sub", "Float", "Float", "Float"), os.path.join(dataset_path, "Float", "add")),
    (Task("sub", "Fraction", "Fraction", "Fraction"), os.path.join(dataset_path, "Fraction", "add")),
    (Task("sub", "ScientificNotation", "ScientificNotation", "ScientificNotation"), os.path.join(dataset_path, "ScientificNotation", "add")), # sub can use the same dataset as add, with a swap preprocess in dataset
    
    (Task("max", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "compare")), 
    (Task("max", "Float", "Float", "Float"), os.path.join(dataset_path, "Float", "compare")),
    (Task("max", "Fraction", "Fraction", "Fraction"), os.path.join(dataset_path, "Fraction", "add")), # compare betwee fraction is hard, use a smaller dataset
    (Task("max", "ScientificNotation", "ScientificNotation", "ScientificNotation"), os.path.join(dataset_path, "ScientificNotation", "compare")),
    
    (Task("max_hard", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "compare_harder")), # two integers are generated to share some digits
    (Task("max_hard", "Float", "Float", "Float"), os.path.join(dataset_path, "Float", "compare_harder")), # two floats are more likely to share the int part and some digits 
    (Task("max_hard", "ScientificNotation", "ScientificNotation", "ScientificNotation"), os.path.join(dataset_path, "ScientificNotation", "compare_harder")), # more likely to share the same the exponent and some digits 
    
    (Task("multiply_hard", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "add")),
    (Task("multiply_hard", "Float", "Float", "Float"), os.path.join(dataset_path, "Float", "add")),
    (Task("multiply_hard", "Fraction", "Fraction", "Fraction"), os.path.join(dataset_path, "Fraction", "add")),
    (Task("multiply_hard", "ScientificNotation", "ScientificNotation", "ScientificNotation"), os.path.join(dataset_path, "ScientificNotation", "add")),
    
    (Task("multiply_easy", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "add")),
    (Task("multiply_easy", "Float", "Float", "Float"), os.path.join(dataset_path, "Float", "add")),
    (Task("multiply_easy", "Fraction", "Fraction", "Fraction"), os.path.join(dataset_path, "Fraction", "add")),
    (Task("multiply_easy", "ScientificNotation", "ScientificNotation", "ScientificNotation"), os.path.join(dataset_path, "ScientificNotation", "add")),
    
    (Task("digit_max", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "compare")),
    (Task("digit_max", "Float", "Float", "Float"), os.path.join(dataset_path, "Float", "compare")),
    
    (Task("digit_add", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "compare")),
    (Task("digit_add", "Float", "Float", "Float"), os.path.join(dataset_path, "Float", "compare")),
    
    (Task("get_digit", "Integer", "int", "int"), os.path.join(dataset_path, "Integer", "compare")),
    (Task("get_digit", "Float", "int", "int"), os.path.join(dataset_path, "Float", "compare")),
    
    (Task("length", "Integer", "none", "int"), os.path.join(dataset_path, "Integer", "compare")),
    (Task("length", "Float", "none", "int"), os.path.join(dataset_path, "Float", "compare")),
    
    (Task("truediv", "Integer", "Integer", "Fraction"), os.path.join(dataset_path, "Integer", "add")),
    (Task("truediv", "Fraction", "Fraction", "Fraction"), os.path.join(dataset_path, "Fraction", "add")),
    (Task("floordiv", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "add")),
    (Task("mod", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "add")),
    (Task("mod_easy", "Integer", "Integer", "Integer"), os.path.join(dataset_path, "Integer", "add")),
    
    (Task("to_float", "Fraction", "none", "Float"), os.path.join(dataset_path, "Fraction", "add")),
    (Task("to_float", "ScientificNotation", "none", "Float"), os.path.join(dataset_path, "ScientificNotation", "compare")),
    
    (Task("to_scient", "Integer", "none", "ScientificNotation"), os.path.join(dataset_path, "Integer", "compare")),
    (Task("to_scient", "Float", "none", "ScientificNotation"), os.path.join(dataset_path, "Float", "compare")),
    
    (Task("count", "Integer", "int", "int"), os.path.join(dataset_path, "Integer", "compare")),
    
    (Task("sig", "Integer", "int", "ScientificNotation"), os.path.join(dataset_path, "Integer", "compare")),
    (Task("sig", "Float", "int", "ScientificNotation"), os.path.join(dataset_path, "Float", "compare")),
]