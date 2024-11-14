
def remove_longer_digit_from_tokenizer(tokenizer_path: str) -> dict[int, list[int]]:
    """Return a dict with key as index of longer number tokens to be replaced and value as the list of digit token index."""
    max_length_in_vocab = 3
    
    import tqdm
    import json
    import os
    
    convert_longer_to_digits: dict[int, list[int]] = {}
    
    with open(os.path.join(tokenizer_path, "tokenizer.json")) as f:
        tokenizer_dict = json.load(f)
        vocab: dict[str, int] = tokenizer_dict["model"]["vocab"]
        
        # 1. find 0-9 ids
        digit_tokens_id = [vocab[str(i)] for i in range(10)]
        
        # 2. find all tokens that are longer than 10
        for token_int in tqdm.tqdm(range(10, 10 ** max_length_in_vocab)):
            try:
                convert_longer_to_digits[vocab[str(token_int)]] = [digit_tokens_id[int(s)] for s in str(token_int)]
            except KeyError:
                print(f"Do not find token {str(token_int)} in tokenizer vocab.")
        
        # 3. save the dict
        with open(os.path.join(tokenizer_path, "one_digit_converter.json"), "w") as f:
            json.dump(convert_longer_to_digits, f)
            
    return convert_longer_to_digits