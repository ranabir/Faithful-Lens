
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_results(input_file="results/analysis_results.json"):
    # Load data
    with open(input_file, "r") as f:
        data = json.load(f)
        
    layer_probs = np.array(data["layer_probs"])
    samples = data["samples"]
    
    # 1. Average Probability Plot
    avg_probs = np.mean(layer_probs, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(avg_probs, marker='o', label="Average Token Prob")
    plt.xlabel("Layer Index")
    plt.ylabel("Probability of Correct Answer")
    plt.title("Average Probability of Correct Answer Across Layers (Logit Lens)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/avg_probs.png")
    print("Saved results/avg_probs.png")
    
    # 2. Heatmap
    # We clip probability for better visualization if it's too skewed
    plt.figure(figsize=(12, 10))
    sns.heatmap(layer_probs, cmap="viridis", cbar_kws={'label': 'Probability'})
    plt.xlabel("Layer Index")
    plt.ylabel("Sample Index")
    plt.title("Answer Probability per Layer per Sample")
    plt.savefig("results/heatmap.png")
    print("Saved results/heatmap.png")
    
    # 3. Text Report
    # Check for "early exit" candidates: Prob > 0.5 before layer 15 (assuming 28 layers?)
    # Qwen 2.5 1.5B has 28 layers.
    high_confidence_early = []
    
    for i, probs in enumerate(layer_probs):
        # Check max prob in first half of layers
        early_max = np.max(probs[:14])
        if early_max > 0.1: # Threshold for "suspicious"
            high_confidence_early.append((i, early_max, samples[i]["target_answer"]))
            
    print("\n--- Insight Report ---")
    print(f"Total Samples: {len(samples)}")
    print(f"Average Final Layer Probability: {avg_probs[-1]:.4f}")
    print(f"Potential 'Answer-First' candidates (Prob > 0.1 in first 14 layers): {len(high_confidence_early)}")
    
    for idx, prob, ans in high_confidence_early[:5]:
        print(f"  - Sample {idx}: Answer '{ans}' appeared with prob {prob:.4f} early on.")

if __name__ == "__main__":
    visualize_results()
