
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_aggregate_stats(input_file="results/analysis_results_phase2.json"):
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Results file not found.")
        return

    samples = data["samples"]
    print(f"Analyzing {len(samples)} samples...")
    
    emergence_positions = [] # Relative position (0.0 to 1.0)
    answer_ranks = []
    
    for i, sample in enumerate(samples):
        trace = sample["trace"]
        target = sample["target_answer"]
        total_steps = len(trace)
        
        # Find first "Semantic Emergence"
        # Logic: Gap > -0.1 AND not structural (heuristic: next token not . or ))
        
        emergence_idx = -1
        
        for idx, step in enumerate(trace):
            token = step["generated_token"]
            
            # Check if token contains answer
            if target in token or token.strip() == target:
                final_gap = step["layer_metrics"][-1]["logit_gap"]
                
                # Check structure
                is_structural = False
                if idx + 1 < len(trace):
                    next_token = trace[idx+1]["generated_token"].strip()
                    if next_token.startswith('.') or next_token.startswith(')'):
                        is_structural = True
                
                if final_gap > -0.1 and not is_structural:
                    emergence_idx = idx
                    break # Found the first semantic answer
        
        if emergence_idx != -1:
            emergence_positions.append(emergence_idx / total_steps)
            print(f"Sample {i+1}: Answer emerged at {emergence_idx}/{total_steps} ({emergence_idx/total_steps:.1%})")
            
            # Calculate Faithfulness: Avg Rank of answer BEFORE emergence
            # Ideally this should be high (>> 100)
            pre_emergence_ranks = []
            for step in trace[:emergence_idx]:
                 # Filter out structural mentions (Orange)
                 # If step has high confidence structural answer, we should skip it or treat it as neutral
                 # But our logic says if it's structural, it's NOT the semantic answer.
                 # So rank might be low (0) but it's structure.
                 # We simply take the rank.
                 pre_emergence_ranks.append(step["layer_metrics"][-1]["rank"])
            
            if pre_emergence_ranks:
                avg_rank = np.mean(pre_emergence_ranks)
                print(f"  Faithfulness Score (Avg Rank before emergence): {avg_rank:.1f} (Higher is Better)")
            
            # Find Emergence Layer at the emergence step
            emergence_step_data = trace[emergence_idx]
            emergence_layer = -1
            for layer_metric in emergence_step_data["layer_metrics"]:
                if layer_metric["logit_gap"] > -0.1:
                    emergence_layer = layer_metric["layer"]
                    break
            
            if emergence_layer != -1:
                print(f"  Emergence Layer: {emergence_layer} (0-27)")
        else:
            print(f"Sample {i+1}: Answer never emerged semantically.")
            
    if emergence_positions:
        avg_pos = np.mean(emergence_positions)
        median_pos = np.median(emergence_positions)
        print("-" * 30)
        print(f"Average Emergence Position: {avg_pos:.1%} of reasoning chain")
        print(f"Median Emergence Position:  {median_pos:.1%} of reasoning chain")
        
        # Plot Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(emergence_positions, bins=10, range=(0, 1), edgecolor='black')
        plt.title(f"Histogram of Answer Emergence Position (N={len(emergence_positions)})")
        plt.xlabel("Relative Position in Reasoning Chain (0.0=Start, 1.0=End)")
        plt.ylabel("Count")
        plt.axvline(avg_pos, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_pos:.2f}')
        plt.legend()
        os.makedirs("results/plots", exist_ok=True)
        plt.savefig("results/plots/emergence_histogram.png")
        print("Saved results/plots/emergence_histogram.png")

if __name__ == "__main__":
    compute_aggregate_stats()
