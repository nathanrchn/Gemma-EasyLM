from jax import numpy as jnp
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict
from EasyLM.checkpoint import StreamingCheckpointer

model = StreamingCheckpointer.load_checkpoint("/tmp/mlxu/58f7aa57211149a2954efe2ed978e37b/streaming_train_state_10000")

params = {}

for k, v in flatten_dict(model["params"]["params"], sep=".").items():
    if "embed_tokens" in k:
        params[k.replace("embedding", "weight")] = v.astype(jnp.bfloat16)
    elif "kernel" in k:
        params[k.replace("kernel", "weight")] = jnp.transpose(v.astype(jnp.bfloat16), (1, 0))
    else:
        params[k] = v.astype(jnp.bfloat16)

save_file(params, "gemma-3b/model.safetensors", {"format": "flax"})