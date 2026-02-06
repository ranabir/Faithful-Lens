
import torch
import yaml
import sys
from src.data_loader import load_gsm8k_dataset, filter_single_token_answers
from src.logit_lens import LogitLens

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found.")
        sys.exit(1)

def inspect_layers():
    config = load_config()
    
    # Get inspection params
    target_layers = config.get('inspection', {}).get('layers_to_inspect', [15])
    top_k = config.get('inspection', {}).get('top_k', 5)
    model_name = config['model']['name']
    
    print(f"Inspecting Layers: {target_layers}")
    print(f"Configuration: Model={model_name}, Top-K={top_k}")

    # Setup Device
    device = config['model']['device']
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    lens = LogitLens(model_name=model_name, device=device)
    dataset = load_gsm8k_dataset(split=config['dataset']['split'])
    
    if config['dataset']['filter_single_token']:
         filtered_data = filter_single_token_answers(dataset, lens.tokenizer)
    else:
         filtered_data = dataset
         
    print(f"Scanning first 5 samples...")
    
    for i in range(5):
        sample = filtered_data[i]
        question = sample['question']
        target = sample['extracted_answer']
        
        # Format input
        messages = [
            {"role": "system", "content": "Please reason step by step and put your final answer at the end."},
            {"role": "user", "content": question}
        ]
        text = lens.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Forward pass
        lens.forward(text)
        
        print(f"\n--- Sample {i+1}: Target Answer = '{target}' ---")
        
        for layer_idx in target_layers:
            # Decode Layer at last token
            top_tokens = lens.decode_layer(layer_idx, token_idx=-1, k=top_k)
            
            print(f"  [Layer {layer_idx}] Top predictions:")
            if top_tokens:
                for token, prob in top_tokens:
                    print(f"    '{token}' ({prob:.4f})")
            else:
                print("    (No data for this layer)")
                
            # Also check prob of actual answer
            if 'answer_token_id' in sample:
                target_token_id = sample['answer_token_id']
                probs = lens.get_layer_probs_for_token(target_token_id, token_idx=-1)
                print(f"    > Probability of actual answer '{target}': {probs[layer_idx]:.6f}")

if __name__ == "__main__":
    inspect_layers()
