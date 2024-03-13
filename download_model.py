from transformers import AutoModelForCausalLM

model_name = str(input("Enter the model name: "))

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(model_name, max_shard_size="10GB")