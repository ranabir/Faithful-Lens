
import json
import numpy as np
import os
# from scipy import stats # Module missing, implementing manual bootstrap

def manual_bootstrap_ci(data, n_resamples=1000, alpha=0.95):
    """Compute bootstrap CI manually without scipy."""
    means = []
    n = len(data)
    for _ in range(n_resamples):
        sample = np.random.choice(data, n, replace=True)
        means.append(np.mean(sample)) # Or median if skewed
    
    lower = np.percentile(means, (1 - alpha) / 2 * 100)
    upper = np.percentile(means, (1 + alpha) / 2 * 100)
    return lower, upper

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
    emergence_layers = []   # Layer index (0-28) at emergence
    correct_count = 0
    
    failures = [] # List of (index, reason, trace_snippet)
    
    for i, sample in enumerate(samples):
        trace = sample["trace"]
        target = sample["target_answer"]
        total_steps = len(trace)
        
        full_text = "".join([t["generated_token"] for t in trace])
        is_correct = target in full_text[-20:]
        if is_correct:
            correct_count += 1
        else:
            failures.append({
                "id": i,
                "reason": "Incorrect Answer",
                "text": full_text[-50:],
                "target": target
            })
            continue # Skip incorrect for emergence stats? Or keep? 
                     # Usually keep but mark as "Never Emerged" if it didn't.
                     
        # Find Emergence
        emergence_idx = -1
        found_layer = -1
        
        for idx, step in enumerate(trace):
            # We look for ANY layer where the target is Top-1 (Gap > 0)
            # But we must avoid structural mentions (e.g. "Step 3") which might actually BE the token.
            # If the current token IS the target, and Gap > 0, it's valid, but might be structural.
            # If the current token is NOT the target (e.g. "The"), and Gap > 0 (in Layer 15), it implies latent knowledge.
            
            # 1. Get max gap across layers
            best_layer_gap = -999
            best_layer = -1
            
            for lm in step["layer_metrics"]:
                if lm and lm["logit_gap"] > best_layer_gap:
                    best_layer_gap = lm["logit_gap"]
                    best_layer = lm["layer"]
            
            # Threshold: Gap > -0.1 means it is the Top-1 prediction (or very close)
            # Since Gap = Logit(Target) - Logit(Top1), Gap=0 means Target IS Top1.
            if best_layer_gap > -0.01:
                # 2. Structural/False Positive Filter
                
                # Check A: Is this just the model reading the previous token? 
                # (Not applicable for generation, only prompt processing)
                
                # Check B: Is this an intermediate calculation?
                # Heuristic: Check the *next generated token*.
                # If the current hidden state predicts "3", and the next token IS "3",
                # check if that "3" is followed by an operator.
                
                is_structural = False
                if idx + 1 < len(trace):
                    next_token = trace[idx+1]["generated_token"].strip()
                    # If the prediction matches the next token, we check context
                    if next_token == target: 
                        # Check the token AFTER that
                        if idx + 2 < len(trace):
                            after_next = trace[idx+2]["generated_token"].strip()
                            # Operators, OR unit words like "dollars"? 
                            # If "3 dollars", it might be the answer.
                            # Standard CoT: "So 1+2=3." -> 3 is followed by "."
                            # Answer format: "Answer: 3" -> 3 is followed by <EOS> or similar.
                            
                            # If followed by operator, it is structure.
                            if after_next in ['+', '-', '*', '/', '=', 'times', 'plus', 'minus', '^']:
                                is_structural = True
                
                # Also, we check if the answer emerges VERY early (e.g. step 0).
                # If step 0 gap > -0.01, it might be "Pre-computation" / Leakage.
                # But we treat that as Valid Emergence (Position 0.0).
                
                if not is_structural:
                    emergence_idx = idx
                    found_layer = best_layer
                    break # Found earliest emergence
        
        if emergence_idx != -1:
            emergence_positions.append(emergence_idx / total_steps)
            emergence_layers.append(found_layer)
        else:
            reason = "Never Emerged"
            if not is_correct: reason = "Incorrect & Never Emerged"
            failures.append({
                "id": i,
                "reason": reason,
                "text": full_text[-50:],
                "target": target
            })

    # --- Stats Output ---
    print("\n=== Aggregate Statistics ===")
    acc = correct_count / len(samples)
    print(f"Accuracy: {acc:.1%} ({correct_count}/{len(samples)})")
    
    if emergence_positions:
        avg_pos = np.mean(emergence_positions)
        std_pos = np.std(emergence_positions)
        ci_low, ci_high = manual_bootstrap_ci(emergence_positions)
        
        print(f"\nEmergence Position (Relative):")
        print(f"  Mean: {avg_pos:.1%} ± {std_pos:.1%}")
        print(f"  95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
        
        avg_layer = np.mean(emergence_layers)
        std_layer = np.std(emergence_layers)
        ci_layer_low, ci_layer_high = manual_bootstrap_ci(emergence_layers)
        
        print(f"\nEmergence Layer (0-28):")
        print(f"  Mean: {avg_layer:.1f} ± {std_layer:.1f}")
        print(f"  95% CI: [{ci_layer_low:.1f}, {ci_layer_high:.1f}]")
        
    print("\n=== Top Failures (Sample) ===")
    print("| ID | Reason | Target | Snippet |")
    print("|---|---|---|---|")
    for f in failures[:10]:
        snippet = f['text'].replace('\n', ' ')
        print(f"| {f['id']} | {f['reason']} | {f['target']} | ...{snippet} |")

if __name__ == "__main__":
    compute_aggregate_stats()
