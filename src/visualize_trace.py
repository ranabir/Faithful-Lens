
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def visualize_trace(input_file="results/analysis_results_phase2.json"):
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    with open(input_file, "r") as f:
        data = json.load(f)
        
    samples = data["samples"]
    print(f"Visualizing traces for {len(samples)} samples...")
    
    # We want to visualize a few representative samples
    # Heatmap: X=Time (Tokens), Y=Layer
    # Value: Logit Gap or Log(Rank)
    
    for i, sample in enumerate(samples): # All samples
        trace = sample["trace"]
        target = sample["target_answer"]
        question = sample["question"][:50] + "..."
        
        # Extract matrices
        # shape: [num_layers, num_steps]
        # We need to find num_layers from the first valid metric
        if not trace:
            continue
            
        num_layers = len(trace[0]["layer_metrics"])
        num_steps = len(trace)
        
        rank_matrix = np.zeros((num_layers, num_steps))
        gap_matrix = np.zeros((num_layers, num_steps))
        tokens = []
        
        for t_idx, step in enumerate(trace):
            tokens.append(step["generated_token"])
            for l_metrics in step["layer_metrics"]:
                if l_metrics:
                    l = l_metrics["layer"]
                    rank_matrix[l, t_idx] = l_metrics["rank"]
                    gap_matrix[l, t_idx] = l_metrics["logit_gap"]
                    
        # Plot 1: Logit Gap Heatmap
        plt.figure(figsize=(20, 8))
        sns.heatmap(gap_matrix, cmap="RdBu_r", center=0, cbar_kws={'label': 'Logit Gap (Answer - Top1)'})
        plt.title(f"Sample {i+1}: Answer '{target}' Logit Gap Trace\nQ: {question}")
        plt.xlabel("Reasoning Step (Token)")
        plt.ylabel("Layer Depth")
        
        # Invert Y-axis so Layer 0 is at the BOTTOM
        plt.gca().invert_yaxis()
        
        # X-ticks: show tokens periodically, BUT ALWAYS show the answer token
        step_size = max(1, num_steps // 30)
        
        # Advanced Labeling:
        # ... (Same logic for labels) ...
        
        # Identify indices where token contains the answer
        answer_indices = [idx for idx, t in enumerate(tokens) if target in t or t.strip() == target]
        
        # Create ticks
        ticks = []
        labels = []
        for j in range(num_steps):
            if j in answer_indices:
                ticks.append(j + 0.5)
                labels.append(tokens[j])
            elif j % step_size == 0:
                ticks.append(j + 0.5)
                labels.append(tokens[j])
                
        plt.xticks(ticks, labels, rotation=90, fontsize=8)
        
        # Color the answer labels red
        ax = plt.gca()
        for label, tick_pos in zip(ax.get_xticklabels(), ticks):
            idx = int(tick_pos - 0.5)
            text = label.get_text()
            
            if target in text or text.strip() == target:
                final_layer_gap = gap_matrix[-1, idx]
                
                # Check for structural context
                is_structural = False
                if idx + 1 < len(tokens):
                    next_token = tokens[idx+1].strip()
                    if next_token.startswith('.') or next_token.startswith(')'):
                        is_structural = True
                
                if final_layer_gap > -0.1: 
                    if is_structural:
                        # ORANGE (Structure)
                        label.set_color('orange') 
                        label.set_style('italic')
                        label.set_text(f"{text} (list)")
                        plt.vlines(x=idx + 0.5, ymin=0, ymax=num_layers, colors='orange', linestyles=':', alpha=0.5)
                    else:
                        # RED (Answer) - Find Emergence Layer
                        label.set_color('red')
                        label.set_fontweight('bold')
                        label.set_text(f"{text} (ANS)")
                        
                        # Scan layers to find where gap becomes > -0.1
                        emergence_layer = 0
                        for l in range(num_layers):
                            if gap_matrix[l, idx] > -0.1:
                                emergence_layer = l
                                break
                        
                        # Draw line only from Emergence to Top
                        plt.vlines(x=idx + 0.5, ymin=emergence_layer, ymax=num_layers, colors='red', linestyles='--', alpha=0.9, linewidth=2)
                        # Optional: Draw a horizontal marker at emergence
                        plt.hlines(y=emergence_layer, xmin=idx, xmax=idx+1, colors='red', linestyles='-', linewidth=2)
                else:
                    # GRAY (Low Confidence)
                    label.set_color('gray')
                    label.set_style('italic')
                    label.set_text(f"{text} (seq)")
                    plt.vlines(x=idx + 0.5, ymin=0, ymax=num_layers, colors='gray', linestyles=':', alpha=0.3)

        
        plt.tight_layout()
        os.makedirs("results/plots", exist_ok=True)
        plt.savefig(f"results/plots/trace_sample_{i+1}_gap.png")
        print(f"Saved results/plots/trace_sample_{i+1}_gap.png")
        plt.close()

if __name__ == "__main__":
    visualize_trace()
