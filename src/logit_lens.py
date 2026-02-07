import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class LogitLens:
    def __init__(self, model_name="Qwen/Qwen2.5-Math-1.5B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device != "cuda" and self.model.device.type != "cuda":
             self.model.to(device)

        self.hook_handles = []
        self.activations = {}
        self.num_layers = self.model.config.num_hidden_layers
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward hooks on each layer to capture the residual stream (output of the layer).
        """
        # Clear existing hooks if any
        self.clear_hooks()
        
        # Specific to Qwen/Llama architectures generally. 
        # Usually model.model.layers[i]
        layers = self.model.model.layers
        
        for i, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._get_hook(i))
            self.hook_handles.append(handle)

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            # output is typically a tuple (hidden_state, present_key_value, ...)
            # We want the hidden state [batch, seq_len, search_dim]
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            
            # Detach and move to CPU to save GPU memory if needed, 
            # or keep on GPU for faster decoding if VRAM assumes.
            # Storing on CPU for safety now.
            self.activations[layer_idx] = hidden_state.detach()
        return hook

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.activations = {}

    def forward(self, input_text):
        """
        Runs a forward pass and captures activations.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        self.activations = {} # Reset
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def decode_layer(self, layer_idx, token_idx=-1, k=10):
        """
        Decodes the hidden state at a specific layer and token position using the unembedding matrix.
        
        Args:
            layer_idx (int): The layer index (0 to num_layers-1).
            token_idx (int): The token index to decode (default -1 for the last token).
            k (int): Top-k tokens to return.
            
        Returns:
            List of (token_str, probability) tuples.
        """
        if layer_idx not in self.activations:
            return None
        
        # Get hidden state: [batch, seq, dim] -> [1, seq, dim] usually
        hidden_state = self.activations[layer_idx]
        
        # Select specific token vector
        # hidden_state[0, token_idx, :] -> [dim]
        vector = hidden_state[0, token_idx, :]
        
        # Project to vocabulary: vector @ lm_head.weight.T
        # Note: Logits = Hidden @ W_U. usually in model.lm_head
        # Normalization (LayerNorm) before lm_head?
        # Qwen/Llama usually have a final layernorm (model.norm) before lm_head.
        # Logit Lens typically implies "decode the residual DIRECTLY", 
        # but often applying the final LN makes it more interpretable.
        # We will apply final LayerNorm first as is standard in modern Logit Lens practices.
        
        final_norm = self.model.model.norm
        vector_normed = final_norm(vector) # Apply final RMSNorm/LayerNorm
        
        logits = self.model.lm_head(vector_normed)
        probs = torch.softmax(logits, dim=-1)
        
        top_probs, top_ids = torch.topk(probs, k)
        
        results = []
        for p, i in zip(top_probs, top_ids):
            token = self.tokenizer.decode(i.item())
            results.append((token, p.item()))
            
        return results

    def get_metrics_for_token(self, target_token_str, token_idx=-1):
        """
        Returns robust metrics for a target answer string across all layers.
        Handles token canonicalization (e.g. " 72", "72").
        
        Args:
            target_token_str (str): The answer string (e.g. "72").
            token_idx (int): Position to probe.
            
        Returns:
            list of dict: Metrics per layer [{'prob': ..., 'rank': ..., 'logit_gap': ...}]
        """
        # Canonicalize: Create variants
        variants = [
            target_token_str,
            " " + target_token_str,
            target_token_str.strip(),
            "\n" + target_token_str.strip()
        ]
        # Filter unique and encode
        variant_ids = set()
        for v in variants:
            # We want the ID of the *single token* that represents this string.
            # If it tokenizes to multiple, we skip it for now (or take first).
            # This method assumes the answer IS a single token in some form.
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                variant_ids.add(ids[0])
        
        if not variant_ids:
            return None # Answer cannot be represented as single token

        metrics_per_layer = []
        final_norm = self.model.model.norm

        for i in range(self.num_layers):
            if i in self.activations:
                hidden = self.activations[i][0, token_idx, :]
                hidden_normed = final_norm(hidden) # Ensure LN is applied
                logits = self.model.lm_head(hidden_normed)
                probs = torch.softmax(logits, dim=-1)
                
                # Get metrics for the BEST variant at this layer
                best_prob = -1.0
                best_rank = float('inf')
                best_logit = -float('inf')
                
                # Global top-1 logit for gap
                top_logits, top_indices = torch.topk(logits, k=1)
                top_1_logit = top_logits[0].item()
                
                for vid in variant_ids:
                    p = probs[vid].item()
                    l = logits[vid].item()
                    
                    # Calculate Rank efficiently? 
                    # (logits > l).sum() is rank (0-indexed)
                    r = (logits > l).sum().item()
                    
                    if p > best_prob:
                        best_prob = p
                        best_rank = r
                        best_logit = l
                
                metrics_per_layer.append({
                    "layer": i,
                    "prob": best_prob,
                    "rank": best_rank,
                    "logit": best_logit,
                    "logit_gap": best_logit - top_1_logit
                })
            else:
                 metrics_per_layer.append(None)
                 
        return metrics_per_layer

    def get_layer_probs_for_token(self, target_token_id, token_idx=-1):
        """
        Legacy method. Use get_metrics_for_token for robust analysis.
        """
        layer_probs = []
        final_norm = self.model.model.norm

        for i in range(self.num_layers):
            if i in self.activations:
                hidden = self.activations[i][0, token_idx, :]
                hidden_normed = final_norm(hidden)
                logits = self.model.lm_head(hidden_normed)
                probs = torch.softmax(logits, dim=-1)
                target_prob = probs[target_token_id].item()
                layer_probs.append(target_prob)
            else:
                layer_probs.append(0.0)
        return layer_probs
