import re
from datasets import load_dataset

def load_gsm8k_dataset(split="test"):
    """
    Loads the GSM8K dataset.
    Args:
        split (str): 'train' or 'test'.
    Returns:
        Dataset object.
    """
    dataset = load_dataset("gsm8k", "main", split=split)
    return dataset

def extract_answer(solution):
    """
    Extracts the numerical answer from the GSM8K solution string.
    The answer is typically after '#### '.
    """
    if "####" not in solution:
        return None
    return solution.split("####")[-1].strip()

def filter_single_token_answers(dataset, tokenizer):
    """
    Filters the dataset to keep only examples where the answer 
    tokenizes to a single token.
    """
    filtered_data = []
    for item in dataset:
        answer = extract_answer(item['answer'])
        if answer is None:
            continue
        
        # Check if answer is single token
        # Note: We add a leading space or whatever the tokenizer expects usually,
        # but for numbers often they are single tokens.
        # This is a heuristic; we might need more robust checks depending on the tokenizer.
        tokens = tokenizer.encode(answer, add_special_tokens=False)
        if len(tokens) == 1:
            filtered_data.append({
                'question': item['question'],
                'answer': item['answer'],
                'extracted_answer': answer,
                'answer_token_id': tokens[0]
            })
            
    return filtered_data
