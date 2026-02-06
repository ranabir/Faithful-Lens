
import sys
import torch
from src.data_loader import load_gsm8k_dataset, filter_single_token_answers
from src.analysis import run_analysis

def main():
    print("Loading data...")
    tokenizer_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    dataset = load_gsm8k_dataset(split="test")
    filtered_data = filter_single_token_answers(dataset, tokenizer)
    
    if not filtered_data:
        print("No single-token answers found!")
        return

    print(f"Found {len(filtered_data)} single-token answer examples.")
    print("Running analysis on 2 samples...")
    
    try:
        results = run_analysis(
            model_name=tokenizer_name,
            dataset=filtered_data,
            num_samples=2,
            device="cpu" # Use CPU for safety in test, change to cuda if sure
        )
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Analysis complete.")
    print("Sample 1 Question:", results["samples"][0]["question"])
    print("Sample 1 Target:", results["samples"][0]["target_answer"])
    print("Sample 1 Layer Probs (Top 3 layers):", results["layer_probs"][0][:3])
    print("Sample 1 Layer Probs (Last 3 layers):", results["layer_probs"][0][-3:])
    
if __name__ == "__main__":
    main()
