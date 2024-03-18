import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepnetguy/gemma-126"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.save_pretrained(model_name, max_shard_size="10GB")

tokenizer = AutoTokenizer.from_pretrained("NousResearch/gemma-2b-it-tokenizer")
tokenizer.save_pretrained(model_name)