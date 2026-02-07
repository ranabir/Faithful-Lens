import torch
import numpy as np
import os
import json
from tqdm import tqdm
from .logit_lens import LogitLens

def run_analysis_phase2(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct", 
    dataset=None, 
    num_samples=10, 
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Runs Phase 2 analysis: Tracing answer probability/rank across the entire generated reasoning chain.
    """
    lens = LogitLens(model_name=model_name, device=device)
    
    results = {
        "samples": []
    }
    
    count = 0
    for item in tqdm(dataset):
        if count >= num_samples:
            break
            
        question = item['question']
        target_answer = item['extracted_answer']
        
        # Format input
        messages = [
            {"role": "system", "content": "Please reason step by step and put your final answer at the end."},
            {"role": "user", "content": question}
        ]
        prompt_text = lens.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 1. Generate the FULL reasoning chain first
        inputs = lens.tokenizer(prompt_text, return_tensors="pt").to(lens.device)
        with torch.no_grad():
            output_ids = lens.model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False # Greedy decoding for determinism
            )
        
        # 2. Trace the answer at every step
        # generated_ids excludes prompt
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        full_text = lens.tokenizer.decode(output_ids[0])
        generated_text = lens.tokenizer.decode(generated_ids)
        
        # We need to run forward pass on the full sequence and check activations at each step
        # Optimization: Run once on full sequence, then look at indices corresponding to generation
        lens.activations = {} # Clear old hooks
        with torch.no_grad():
            lens.model(output_ids) # Forward pass on (Prompt + Generation)
            
        # lens.activations now contains [1, seq_len, dim]
        # We want to inspect positions starting from len(prompt)-1 (last token of prompt)
        # to len(full)-2 (last token of generation, predicting the EOS or next)
        
        prompt_len = inputs.input_ids.shape[1]
        total_len = output_ids.shape[1]
        
        sample_trace = []
        
        # Iterate through relevant positions
        # At position i, the model predicts token at i+1
        # Range: prompt_len-1 TO total_len-2
        
        for pos in range(prompt_len - 1, total_len - 1):
            # What token was actually generated next?
            next_token_id = output_ids[0, pos+1].item()
            next_token_str = lens.tokenizer.decode(next_token_id)
            
            # Get metrics for the FINAL ANSWER at this position
            metrics = lens.get_metrics_for_token(target_answer, token_idx=pos)
            
            if metrics:
                # We only store the "best" layer or average? Or all layers?
                # Storing all layers for 512 tokens * 28 layers is big but manageable for 50 samples.
                # Let's store compact data.
                
                step_data = {
                    "position": pos - (prompt_len - 1), # Relative step (0 = end of prompt)
                    "generated_token": next_token_str,
                    "layer_metrics": metrics # List of dicts
                }
                sample_trace.append(step_data)
        
        results["samples"].append({
            "question": question,
            "target_answer": target_answer,
            "trace": sample_trace
        })
        
        count += 1
        
        # Incremental Save (Every 5 samples)
        if count % 5 == 0:
            print(f"Saving intermediate results ({count} samples)...")
            temp_output = "results/analysis_results_phase2.json"
            os.makedirs("results", exist_ok=True)
            with open(temp_output, "w") as f:
                json.dump(results, f, indent=2)
        
    return results

# Keep Phase 1 function for compatibility if needed, or redirect
def run_analysis(phase=1, **kwargs):
    if phase == 2:
        return run_analysis_phase2(**kwargs)
    else:
        # Import original phase 1 logic or keep it here 
        # (Assuming we replaced the file, we should have kept Phase 1 too or moved it)
        # For now, let's assume we proceed with Phase 2 mostly.
        pass
