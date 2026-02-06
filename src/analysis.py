
import torch
import numpy as np
from tqdm import tqdm
from .logit_lens import LogitLens

def run_analysis(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct", 
    dataset=None, 
    num_samples=10, 
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Runs the logit lens analysis on a subset of the dataset.
    
    Args:
        model_name (str): Model identifier.
        dataset (list): List of dicts with 'question', 'answer', 'answer_token_id'.
        num_samples (int): Number of samples to process.
        device (str): Device to run on.
        
    Returns:
        dict: Results containing 'layer_probs' for each sample.
    """
    
    lens = LogitLens(model_name=model_name, device=device)
    
    results = {
        "samples": [],
        "layer_probs": [] # List of [num_layers] lists, one per sample
    }
    
    count = 0
    # Simple prompt template for Qwen-Math, check if it's chat model or base.
    # The id suggests Instruct, so we should use chat template if possible, 
    # but for raw reasoning trace, maybe direct question is okay.
    # Let's try to use the chat template from tokenizer often stored in apply_chat_template
    
    for item in tqdm(dataset):
        if count >= num_samples:
            break
            
        question = item['question']
        target_token_id = item['answer_token_id']
        
        # Format input
        messages = [
            {"role": "system", "content": "Please reason step by step and put your final answer at the end."},
            {"role": "user", "content": question}
        ]
        text = lens.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # We need to generate the reasoning chain First to see where the answer *should* be.
        # But for "Answer-First" bias, we often look at the *last token of the prompt* 
        # or the *first token of the completion*.
        # The hypothesis is: Does the answer appear in the residual stream at the END OF THE PROMPT?
        # Or at the FIRST few tokens of generation?
        # Let's check the PREDICTION at the END OF THE PROMPT (before any reasoning).
        
        # Forward pass on the prompt only
        lens.forward(text)
        
        # Check rank of the target answer at the last token of the input
        # The 'logits' returned by forward are [batch, seq, vocab]
        # logic_lens stores activations for the prompt pass.
        # We look at the last position (-1).
        
        probs = lens.get_layer_probs_for_token(target_token_id, token_idx=-1)
        
        results["samples"].append({
            "question": question,
            "target_answer": item['extracted_answer'],
            "target_token_id": target_token_id
        })
        results["layer_probs"].append(probs)
        
        count += 1
        
    return results
