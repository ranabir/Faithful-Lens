"""
Script to run Causal Control Experiments.
Verifies that the "Computation Hub" found in tracing is causally necessary and distinct from
random noise or irrelevant tokens.

Experiments:
1. Random Noise: Patching from unrelated prompt (Should destroy performance).
2. Irrelevant Token: Patching to a non-hub token (Should have 0 effect).
3. Hub Restoration: Patching clean state to corrupt run at the Hub (Should restore, but maybe not fully if distributed).
4. Sanity: Identity patch (Should be 0).
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import re

# Hardcoded for simplicity/reproducibility
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model.eval()

def perturb_question(question):
    """
    Finds first integer, adds 11 to it.
    """
    match = re.search(r"\b(\d+)\b", question)
    if not match: return None
    original_num_str = match.group(1)
    original_num = int(original_num_str)
    corrupt_num = original_num + 11
    corrupt_q = question.replace(original_num_str, str(corrupt_num), 1)
    return corrupt_q

def get_activations(text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    stored_acts = {}
    def hook_fn(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            stored_acts[layer_idx] = h.detach().cpu()
        return hook
    
    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(hook_fn(i)))
    
    with torch.no_grad():
        _ = model(**inputs)
        
    for h in handles:
        h.remove()
        
    return stored_acts

def patch_and_measure(corrupt_text, source_acts, target_token_id, layer_idx, token_idx):
    inputs = tokenizer(corrupt_text, return_tensors="pt").to(DEVICE)
    
    # Base Logit
    with torch.no_grad():
        out_base = model(**inputs)
        logit_base = out_base.logits[0, -1, target_token_id].item()
    
    # Patch
    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
            
        src = source_acts[layer_idx].to(DEVICE)
        
        # 3D [b, s, d]
        if len(act.shape) == 3:
            # Check bounds
            t_idx = min(token_idx, act.shape[1]-1)
            src_t_idx = min(token_idx, src.shape[1]-1)
            
            act[:, t_idx, :] = src[:, src_t_idx, :]
            
        # 2D [s, d]
        elif len(act.shape) == 2:
            t_idx = min(token_idx, act.shape[0]-1)
            src_t_idx = min(token_idx, src.shape[0]-1)
            
            act[t_idx, :] = src[src_t_idx, :]
            
        return (act,) + output[1:] if isinstance(output, tuple) else act

    layer_module = model.model.layers[layer_idx]
    handle = layer_module.register_forward_hook(patch_hook)
    
    try:
        with torch.no_grad():
            out_patch = model(**inputs)
            logit_patch = out_patch.logits[0, -1, target_token_id].item()
    finally:
        handle.remove()
        
    return logit_patch - logit_base

import matplotlib.pyplot as plt
import os

def plot_sample_controls(sample_idx, effects, title_suffix):
    """
    Plots a bar chart for a single sample's control results.
    effects: dict with keys 'hub', 'irr', 'rnd', 'san' (values are floats)
    """
    labels = ['Hub (Clean)', 'Irrelevant', 'Random (Noise)', 'Sanity (Id)']
    values = [effects['hub'], effects['irr'], effects['rnd'], effects['san']]
    colors = ['blue', 'gray', 'red', 'green']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.7)
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("Logit Difference (Causal Effect)")
    plt.title(f"Control Experiments - Sample {sample_idx}\nTarget: '{title_suffix}'")
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", va='bottom' if yval < 0 else 'bottom', ha='center', fontweight='bold')
        
    os.makedirs("results/causal", exist_ok=True)
    path = f"results/causal/control_sample_{sample_idx}.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved plot to {path}")

def run_controls():
    print("Loading GSM8K...")
    # Load just a few to find 10 valid ones
    try:
        ds = load_dataset("gsm8k", "main", split="test")
    except:
        print("Dataset load failed, using fallback list.")
        return

    random_prompt = "The capital of France is Paris. The capital of Germany is Berlin."
    random_acts = get_activations(random_prompt)
    
    results = {"hub": [], "irrelevant": [], "random": [], "sanity": []}
    
    TEMPLATE = "Question: {question}\nAnswer using only the final number value.\nAnswer:"
    
    valid_count = 0
    target_count = 50
    
    for i in range(len(ds)):
        if valid_count >= target_count: break
        
        row = ds[i]
        q_raw = row['question']
        
        # 1. Perturb
        q_corrupt = perturb_question(q_raw)
        if not q_corrupt: continue
        
        clean_prompt = TEMPLATE.format(question=q_raw)
        corrupt_prompt = TEMPLATE.format(question=q_corrupt)
        
        # 2. Get Target
        inputs = tokenizer(clean_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        generated_ids = out[0, inputs.input_ids.shape[1]:]
        
        # Find first non-space token
        target_id = None
        target_str = ""
        for tid in generated_ids:
            s = tokenizer.decode([tid])
            # Check for digit
            if re.search(r"\d", s):
                target_id = tid.item()
                target_str = s
                break
        
        if target_id is None: continue
        
        print(f"\n--- Sample {valid_count+1}: Target '{target_str}' ---")
        
        
        clean_acts = get_activations(clean_prompt)
        
        # Define Locations
        # Hub: Last token of prompt.
        input_ids = tokenizer.encode(corrupt_prompt)
        last_token_idx = len(input_ids) - 1
        hub_layer = 15
        
        # Irrelevant: Token 5
        irr_idx = 5
        
        # Measure
        eff_hub = patch_and_measure(corrupt_prompt, clean_acts, target_id, hub_layer, last_token_idx)
        eff_irr = patch_and_measure(corrupt_prompt, clean_acts, target_id, hub_layer, irr_idx)
        eff_rnd = patch_and_measure(corrupt_prompt, random_acts, target_id, hub_layer, last_token_idx)
        eff_san = patch_and_measure(clean_prompt, clean_acts, target_id, hub_layer, last_token_idx)
        
        print(f"  Hub: {eff_hub:.4f} | Irr: {eff_irr:.4f} | Rnd: {eff_rnd:.4f} | San: {eff_san:.4f}")
        
        results["hub"].append(eff_hub)
        results["irrelevant"].append(eff_irr)
        results["random"].append(eff_rnd)
        results["sanity"].append(eff_san)
        
        # Plot first 2 samples
        if valid_count < 2:
            plot_sample_controls(valid_count, {
                'hub': eff_hub, 'irr': eff_irr, 'rnd': eff_rnd, 'san': eff_san
            }, target_str)
            
        valid_count += 1

    print("\n=== Summary (N=50) ===")
    mean_hub = np.mean(results['hub'])
    std_hub = np.std(results['hub'])
    mean_irr = np.mean(results['irrelevant'])
    std_irr = np.std(results['irrelevant'])
    mean_rnd = np.mean(results['random'])
    std_rnd = np.std(results['random'])
    mean_san = np.mean(results['sanity'])
    std_san = np.std(results['sanity'])
    
    print(f"Hub Restoration: {mean_hub:.4f} +/- {std_hub:.4f}")
    print(f"Irrelevant Tok:  {mean_irr:.4f} +/- {std_irr:.4f}")
    print(f"Random Source:   {mean_rnd:.4f} +/- {std_rnd:.4f}")
    print(f"Sanity Check:    {mean_san:.4f} +/- {std_san:.4f}")
    
    # Plot Aggregate
    labels = ['Hub (Clean)', 'Irrelevant', 'Random (Noise)', 'Sanity (Id)']
    means = [mean_hub, mean_irr, mean_rnd, mean_san]
    stds = [std_hub, std_irr, std_rnd, std_san]
    colors = ['blue', 'gray', 'red', 'green']
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("Logit Difference (Causal Effect)")
    plt.title(f"Aggregate Control Experiments (N=50)\nMean causal effect on target logit")
    
    # Save
    os.makedirs("results/causal", exist_ok=True)
    plt.savefig("results/causal/control_aggregate_n50.png")
    print("Saved aggregate plot to results/causal/control_aggregate_n50.png")

if __name__ == "__main__":
    run_controls()
