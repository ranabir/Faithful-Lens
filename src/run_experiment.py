
import json
import torch
import yaml
import os
from src.data_loader import load_gsm8k_dataset, filter_single_token_answers
from src.analysis import run_analysis
from src.logit_lens import LogitLens

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    print(f"Loading experiment with config: {config}")
    
    # Setup Device
    device = config['model']['device']
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")
    
    # Setup Model
    tokenizer_name = config['model']['name']
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    # Load Data
    print("Loading data...")
    dataset = load_gsm8k_dataset(split=config['dataset']['split'])
    
    if config['dataset']['filter_single_token']:
        filtered_data = filter_single_token_answers(dataset, tokenizer)
        print(f"Filtered to {len(filtered_data)} single-token answer samples.")
    else:
        filtered_data = dataset
        
    # Limit samples
    num_samples = config['dataset']['num_samples']
    if num_samples > len(filtered_data):
        num_samples = len(filtered_data)
    
    print(f"Running analysis on {num_samples} samples...")
    
    # Run Analysis
    try:
        results = run_analysis(
            model_name=tokenizer_name,
            dataset=filtered_data,
            num_samples=num_samples,
            device=device
        )
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Analysis complete.")
    
    # Save results
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "analysis_results.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {output_file}")
    
if __name__ == "__main__":
    main()
