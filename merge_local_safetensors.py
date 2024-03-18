from tqdm import tqdm
from safetensors.flax import load_file, save_file

model = load_file("silvainrichou/gemma-2b-012/model.safetensors")

merged_model = {}

merged_model["model.embed_tokens.weight"] = model["model.embed_tokens.weight"]

for i in tqdm(range(12)):
    merged_model[f"model.layers.{i}.self_attn.q_proj.weight"] = model[f"model.layers.{i}.self_attn.q_proj.weight"]
    merged_model[f"model.layers.{i}.self_attn.k_proj.weight"] = model[f"model.layers.{i}.self_attn.k_proj.weight"]
    merged_model[f"model.layers.{i}.self_attn.v_proj.weight"] = model[f"model.layers.{i}.self_attn.v_proj.weight"]
    merged_model[f"model.layers.{i}.self_attn.o_proj.weight"] = model[f"model.layers.{i}.self_attn.o_proj.weight"]
    merged_model[f"model.layers.{i}.mlp.gate_proj.weight"] = model[f"model.layers.{i}.mlp.gate_proj.weight"]
    merged_model[f"model.layers.{i}.mlp.up_proj.weight"] = model[f"model.layers.{i}.mlp.up_proj.weight"]
    merged_model[f"model.layers.{i}.mlp.down_proj.weight"] = model[f"model.layers.{i}.mlp.down_proj.weight"]
    merged_model[f"model.layers.{i}.input_layernorm.weight"] = model[f"model.layers.{i}.input_layernorm.weight"]
    merged_model[f"model.layers.{i}.post_attention_layernorm.weight"] = model[f"model.layers.{i}.post_attention_layernorm.weight"]

for i in tqdm(range(6, 18)):
    merged_model[f"model.layers.{i + 6}.self_attn.q_proj.weight"] = model[f"model.layers.{i}.self_attn.q_proj.weight"]
    merged_model[f"model.layers.{i + 6}.self_attn.k_proj.weight"] = model[f"model.layers.{i}.self_attn.k_proj.weight"]
    merged_model[f"model.layers.{i + 6}.self_attn.o_proj.weight"] = model[f"model.layers.{i}.self_attn.o_proj.weight"]
    merged_model[f"model.layers.{i + 6}.self_attn.v_proj.weight"] = model[f"model.layers.{i}.self_attn.v_proj.weight"]
    merged_model[f"model.layers.{i + 6}.mlp.gate_proj.weight"] = model[f"model.layers.{i}.mlp.gate_proj.weight"]
    merged_model[f"model.layers.{i + 6}.mlp.up_proj.weight"] = model[f"model.layers.{i}.mlp.up_proj.weight"]
    merged_model[f"model.layers.{i + 6}.mlp.down_proj.weight"] = model[f"model.layers.{i}.mlp.down_proj.weight"]
    merged_model[f"model.layers.{i + 6}.input_layernorm.weight"] = model[f"model.layers.{i}.input_layernorm.weight"]
    merged_model[f"model.layers.{i + 6}.post_attention_layernorm.weight"] = model[f"model.layers.{i}.post_attention_layernorm.weight"]

merged_model["model.norm.weight"] = model["model.norm.weight"]

save_file(merged_model, "silvainrichou/gemma-2b-012/merged_model.safetensors")