import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

# Load Config
import yaml
# Check if config.yaml is in current dir or parent
if os.path.exists("config.yaml"):
    config_path = "config.yaml"
elif os.path.exists("../config.yaml"):
    config_path = "../config.yaml"
else:
    # Assume root execution
    config_path = "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model.eval()

class CausalTracer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.activations = {}

    def get_activations(self, text):
        """Runs forward pass and captures all layer activations."""
        self.activations = {}
        
        def get_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                self.activations[layer_idx] = h.detach().cpu()
            return hook

        hooks = []
        for i, layer in enumerate(self.model.model.layers):
            hooks.append(layer.register_forward_hook(get_hook(i)))

        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        for h in hooks:
            h.remove()
            
        return self.activations, outputs.logits

    def trace_with_patch(self, clean_text, corrupt_text, target_token_id):
        print(f"Tracing: Clean='{clean_text}' | Corrupt='{corrupt_text}'")
        
        clean_acts, _ = self.get_activations(clean_text)
        
        corrupt_inputs = self.tokenizer(corrupt_text, return_tensors="pt").to(DEVICE)
        seq_len = corrupt_inputs.input_ids.shape[1]
        clean_len = clean_acts[0].shape[1]
        
        print(f"Seq Len Clean: {clean_len} | Seq Len Corrupt: {seq_len}")
        if seq_len != clean_len:
            print("WARNING: Sequence lengths mismatch! Truncating to shorter length for patching.")
            seq_len = min(seq_len, clean_len)
        num_layers = len(self.model.model.layers)
        
        heatmap = np.zeros((num_layers, seq_len))

        # Base Run (Corrupt)
        with torch.no_grad():
            base_out = self.model(**corrupt_inputs)
            base_logits = base_out.logits[0, -1]
            base_target_logit = base_logits[target_token_id].item()
            print(f"Base Logit of target ({target_token_id}): {base_target_logit:.4f}")

        for layer_idx in tqdm(range(num_layers), desc="Scanning Layers"):
            for token_idx in range(seq_len):
                
                def patch_hook(module, input, output):
                    if isinstance(output, tuple):
                        act = output[0]
                    else:
                        act = output
                    
                    # Both act and clean_acts[layer_idx] should be [1, seq, dim]
                    target_act = clean_acts[layer_idx].to(act.device)
                    
                    # Patch: replace [:, token_idx, :] with target
                    act[:, token_idx, :] = target_act[:, token_idx, :]

                    if isinstance(output, tuple):
                        return (act,) + output[1:]
                    else:
                        return act

                hook_handle = self.model.model.layers[layer_idx].register_forward_hook(patch_hook)
                
                try:
                    with torch.no_grad():
                        out = self.model(**corrupt_inputs)
                        patched_logit = out.logits[0, -1, target_token_id].item()
                        # Metric: Logit Difference (Patched - Base)
                        # Positive means we increased the probability of the clean answer
                        heatmap[layer_idx, token_idx] = patched_logit - base_target_logit
                finally:
                    hook_handle.remove()

        return heatmap, base_target_logit

def plot_heatmap(heatmap, tokens, title, save_path):
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap, xticklabels=tokens, cmap="viridis")
    plt.title(title)
    plt.xlabel("Token Position")
    plt.ylabel("Layer")
    plt.tight_layout()
    os.makedirs("results/causal", exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved heatmap to {save_path}")
    plt.close()

if __name__ == "__main__":
    tracer = CausalTracer(model, tokenizer)
    
    # Example: Simple Arithmetic Subject
    # Clean: "John has 5 apples." -> Target "5"
    # Corrupt: "John has 9 apples." -> Target "5" (We want to restore "5")
    
    clean_prompt = "Question: John has 5 apples and buys 2 more. How many apples does he have? Answer:"
    corrupt_prompt = "Question: John has 9 apples and buys 2 more. How many apples does he have? Answer:"
    
    target_answer = " 7" 
    target_id = tokenizer.encode(target_answer)[0] # e.g. " 7"
    
    # Note: Tokens must align! 5 vs 9 is 1 token diff.
    heatmap, base_prob = tracer.trace_with_patch(clean_prompt, corrupt_prompt, target_id)
    
    # Re-tokenize locally for analysis
    start_inputs = tokenizer(corrupt_prompt, return_tensors="pt")
    tokens = [tokenizer.decode([t]) for t in start_inputs.input_ids[0]]
    
    # Analyze Top Effects
    flat_indices = np.argsort(heatmap.flatten())[::-1][:10]
    print("\nTOP CAUSAL SITES (Where patching restored '7'):")
    for idx in flat_indices:
        layer = idx // heatmap.shape[1]
        token_eval = idx % heatmap.shape[1]
        token_str = tokens[token_eval]
        prob = heatmap[layer, token_eval]
        print(f"Layer {layer:02d} | Token '{token_str}' (Pos {token_eval}) | Prob: {prob:.4f}")

    plot_heatmap(heatmap, tokens, 
                 f"Causal Trace (Restoring '{target_answer}')\nBase Prob: {base_prob:.4f}", 
                 "results/causal/simple_trace.png")
