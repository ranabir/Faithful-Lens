
import json
import sys

def explain():
    try:
        with open("results/analysis_results_phase2.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Results file not found.")
        return

    sample = data["samples"][0]
    question = sample["question"]
    target = sample["target_answer"]
    trace = sample["trace"]
    
    print(f"Question: {question}")
    print(f"Target Answer: '{target}'")
    print("-" * 50)
    
    full_text = ""
    for step in trace:
        token = step["generated_token"]
        full_text += token
        
        # Check metrics at final layer (Layer 27)
        metrics = step["layer_metrics"][-1]
        rank = metrics["rank"]
        gap = metrics["logit_gap"]
        
        # Heuristic: Print if rank < 100 OR if token is related to answer
        if rank < 10 or target in token:
            print(f"Step {step['position']:3d} | Token: '{token}' | Rank: {rank:5d} | Logit Gap: {gap:.2f}")

    print("-" * 50)
    print("Full Text generated:")
    print(full_text)

if __name__ == "__main__":
    explain()
