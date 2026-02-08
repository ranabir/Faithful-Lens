import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
DEVICE = "cpu"

print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)

text = "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nLet's think step by step.\nAnswer:"

inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print("Generated:", tokenizer.decode(output[0]))
