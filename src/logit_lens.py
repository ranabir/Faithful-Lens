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

    def get_layer_probs_for_token(self, target_token_id, token_idx=-1):
        """
        Returns the probability of a specific target token across all layers.
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
