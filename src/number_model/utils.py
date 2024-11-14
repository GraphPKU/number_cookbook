from typing import Literal, get_args
from numbers_class import Domain, NumberBasic
import multiprocessing

def change_model_config(config, tokenizer):
    """Modify the original config of Llama model."""
    config.vocab_size = tokenizer.num_token
    config.return_dict_in_generate = False
    config.torch_dtype = "bfloat16"
    config.pad_token_id = tokenizer.addition_token["pad_token"]
    config.bos_token_id = tokenizer.addition_token["bot_token"]
    config.eos_token_id = tokenizer.addition_token["eot_token"]
    return config

def readable_model_size(model_size: int) -> str:
    if model_size > 1e9:
        return f"{model_size/1e9:.0f}B"
    elif model_size > 1e6:
        return f"{model_size/1e6:.0f}M"
    elif model_size > 1e3:
        return f"{model_size/1e3:.0f}K"
    else:
        return f"{model_size}"
    
def process_batch(batch: list[tuple[str, str]], cls, batch_idx: int):
    """Process a batch of data using the specified class and return results with original indices."""
    results = [[cls.from_string(data[0]), cls.from_string(data[1])] for data in batch]
    return batch_idx, results

def parallel_load_string(string_data: list[tuple[str, str]], expect_type: Domain, num_workers: int = 8) -> list:
    cls = NumberBasic.get_subclass(expect_type)

    # Create a pool of workers
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Define the batch size
        batch_size = len(string_data) // num_workers + 1
        batches = [string_data[i:i + batch_size] for i in range(0, len(string_data), batch_size)]
        indices = list(range(len(string_data)))

        # Map the process_batch function to the batches
        results = pool.starmap(process_batch, [(batch, cls, indices[i:i + batch_size]) for i, batch in enumerate(batches)])

    # Flatten the results
    results.sort(key=lambda x: x[0])
    flat_results = sum([result for _, result in results], start=[])
    return flat_results
