
import json

def inspect():
    with open("results/analysis_results_phase2.json", "r") as f:
        data = json.load(f)
    
    sample = data["samples"][0]
    print(f"Target: '{sample['target_answer']}'")
    
    trace = sample["trace"]
    # Print last 5 steps
    print(f"Total steps: {len(trace)}")
    for i in range(max(0, len(trace)-5), len(trace)):
        step = trace[i]
        token = step["generated_token"]
        # Max gap
        best_gap = -999
        for lm in step["layer_metrics"]:
            if lm["logit_gap"] > best_gap:
                best_gap = lm["logit_gap"]
        
        print(f"Step {i}: Token='{token}' | Max Gap={best_gap:.4f}")
        # Print top layer details
        for lm in step["layer_metrics"]:
             if lm["logit_gap"] == best_gap:
                 print(f"   Layer {lm['layer']}: Rank {lm['rank']}, Prob {lm['prob']:.4f}")

if __name__ == "__main__":
    inspect()
