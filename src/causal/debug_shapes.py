import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
DEVICE = "cpu"

print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE) # float32 for cpu safety

text = "Hello world"
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

def debug_hook(i):
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            print(f"Layer {i} Output[0] shape: {h.shape} | Type: {type(h)}")
        else:
            print(f"Layer {i} Output shape: {output.shape} | Type: {type(output)}")
    return hook

for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(debug_hook(i))

print("Running forward pass...")
with torch.no_grad():
    model(**inputs)
