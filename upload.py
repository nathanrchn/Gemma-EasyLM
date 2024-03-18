import torch
from transformers.models.gemma import GemmaForCausalLM, GemmaTokenizer

model_name = "gemma-2b-013"

model = GemmaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.push_to_hub(model_name, private=True, max_shard_size="10GB", token="")

tokenizer = GemmaTokenizer.from_pretrained("NousResearch/gemma-2b-it-tokenizer")
tokenizer.push_to_hub(model_name, token="")