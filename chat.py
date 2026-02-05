import torch
import yaml
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
args = parser.parse_args()

# Load config
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

model_name = config["model"]
user_prompt = config["user_prompt"]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# Format prompt using chat template
messages = [{"role": "user", "content": user_prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize input
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

# Generation config (same as llm_wrapper.py)
generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

# Generate response
with torch.no_grad():
    output_tokens = model.generate(**inputs, generation_config=generation_config)

# Decode only newly generated tokens
input_length = inputs['input_ids'].shape[1]
newly_generated_tokens = output_tokens[0, input_length:]
response = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()

print(response)
