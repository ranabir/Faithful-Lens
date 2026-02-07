
import json

def check_layer_profile():
    with open("results/analysis_results_phase2.json", "r") as f:
        data = json.load(f)
    
    trace = data["samples"][0]["trace"]
    
    # Find the answer step (Step 134 from previous analysis)
    # Or just search for the token "3" that has high rank
    
    target_step = None
    for step in trace:
        if "3" in step["generated_token"]:
             # Check final layer
             if step["layer_metrics"][-1]["logit_gap"] > -0.1:
                 if step["position"] > 100: # Hack to find the later one (Step 134)
                     target_step = step
                     print(f"Found Answer Step at Position {step['position']} (Token: '{step['generated_token']}')")
                     break
    
    if not target_step:
        print("Could not find answer step.")
        return

    print(f"Layer-wise Logit Gap for Step {target_step['position']}:")
    print("Lyr | Gap   | Rank")
    print("----|-------|-----")
    for m in target_step["layer_metrics"]:
        print(f"{m['layer']:3d} | {m['logit_gap']:5.2f} | {m['rank']}")

if __name__ == "__main__":
    check_layer_profile()
