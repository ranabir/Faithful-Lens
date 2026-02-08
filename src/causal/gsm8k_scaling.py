"""
Script to run dense Causal Tracing on GSM8K samples.
Generates a heatmap showing the causal effect of restoration at different layers/tokens.

Usage:
    python -m src.causal.gsm8k_scaling

Settings:
    n_samples: Number of samples to trace (Resulting heatmap is averaged).
    Model: Qwen/Qwen2.5-Math-1.5B-Instruct
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import os

# Import existing tracer
# Assumes running from root as `python -m src.causal.gsm8k_scaling`
from src.causal.trace_experiment import CausalTracer, model, tokenizer, DEVICE

# Load GSM8K Data
from datasets import load_dataset

def perturb_question(question):
    """
    Simple heuristic: Find the first integer in the question and change it.
    Returns (clean_q, corrupt_q, clean_target_str, corrupt_target_str)
    """
    # Regex to find integer
    match = re.search(r"\b(\d+)\b", question)
    if not match:
        return None
    
    original_num_str = match.group(1)
    original_num = int(original_num_str)
    
    # Create corrupt number (e.g. +10, or *2)
    # Avoid 0 or 1 edge cases
    corrupt_num = original_num + 11 
    
    corrupt_q = question.replace(original_num_str, str(corrupt_num), 1)
    
    # We need the MODEL's answer for the clean question to be the target
    # This script assumes we run inference to get the target answer first? 
    # Or we use a known Answer from the dataset?
    
    # For causal tracing, we want to restore the *model's* original answer.
    # So we should generating the answer for Clean Q first.
    
    return corrupt_q

def get_model_answer(question):
    inputs = tokenizer(question, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # We only care about the first token of the answer for the trace target
    # The prompt should end with "Answer:"
    full_out = tokenizer.decode(out[0])
    # Extract part after prompt
    generated = full_out[len(question):]
    return generated

def run_gsm8k_causal_sweep(n_samples=5):
    print(f"Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="test")
    
    tracer = CausalTracer(model, tokenizer)
    
    heatmaps = []
    valid_count = 0
    
    # Custom prompts to force immediate answer (Direct Answer)
    # This avoids CoT "Janet starts with..." which is invariant to the numbers.
    # We want "Answer: 16" vs "Answer: 27".
    TEMPLATE = "Question: {question}\nAnswer using only the final number value.\nAnswer:"
    
    for i in range(len(ds)):
        if valid_count >= n_samples:
            break
        
        row = ds[i]
        q_raw = row['question']
        
        # Perturb
        q_corrupt_raw = perturb_question(q_raw)
        if not q_corrupt_raw:
            continue
            
        clean_prompt = TEMPLATE.format(question=q_raw)
        corrupt_prompt = TEMPLATE.format(question=q_corrupt_raw)
        
        print(f"\n--- Sample {valid_count+1} ---")
        print(f"Clean Q: {q_raw[:50]}...")
        
        # 1. Get Target Answer from Clean Run
        clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(DEVICE)
        
        # Generate a bit more to find the number
        clean_out = model.generate(**clean_inputs, max_new_tokens=5, do_sample=False)
        generated_ids = clean_out[0, clean_inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids)
        print(f"Full Generated: '{generated_text}'")
        
        # Find first token that has a numeric digit
        target_token_id = None
        for tid in generated_ids:
            t_str = tokenizer.decode([tid])
            if re.search(r"\d", t_str): # Contains digit
                target_token_id = tid.item()
                target_token_str = t_str
                break
        
        if target_token_id is None:
            print("Could not find meaningful target token. Skipping.")
            continue

        print(f"Target Token: '{target_token_str}' (ID: {target_token_id})")
        
        # 2. Run Trace
        # Verify prompts align reasonably
        # CausalTracer handles truncation, but let's hope existing perturbation (changing digits) keeps len similar
        try:
            heatmap, base_prob = tracer.trace_with_patch(clean_prompt, corrupt_prompt, target_token_id)
            
            # Normalize heatmap length for aggregation? 
            # Aggregation is hard if lengths differ.
            # Strategy: Just align to the END (Answer start).
            # Or: Interpolate? 
            # Simple approach: Just save the raw heatmaps and plot individually or stack them aligned at end.
            
            # For this MVP, let's just create individual plots or a list.
            # To "Average", we need identical shapes. 
            # We can crop to the last 20 tokens?
            
            rows, cols = heatmap.shape
            CROP_LEN = 20
            if cols >= CROP_LEN:
                 # Take last 20 tokens
                 crop = heatmap[:, -CROP_LEN:]
                 heatmaps.append(crop)
                 valid_count += 1
            else:
                print("Sequence too short, skipping aggregation.")
                
        except Exception as e:
            print(f"Error tracing sample {i}: {e}")
            continue

    if not heatmaps:
        print("No valid traces collected.")
        return

    # Average Heatmap
    avg_heatmap = np.mean(np.array(heatmaps), axis=0)
    
    # Plot
    plt.figure(figsize=(12, 8))
    # X axis tokens are distinct, just use relative position
    plt.title(f"Average Causal Effect (Last 20 Tokens, N={n_samples})")
    sns.heatmap(avg_heatmap, cmap="viridis", xticklabels=range(-20, 0))
    plt.xlabel("Token Position (Relative to Answer)")
    plt.ylabel("Layer")
    plt.savefig("results/causal/gsm8k_average.png")
    print("Saved aggregate heatmap to results/causal/gsm8k_average.png")

if __name__ == "__main__":
    run_gsm8k_causal_sweep(n_samples=10)
