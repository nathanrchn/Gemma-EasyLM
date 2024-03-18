import torch
import numpy as np
from tqdm import tqdm
from safetensors.torch import save_file
from flax.traverse_util import flatten_dict
from EasyLM.checkpoint import StreamingCheckpointer

model = StreamingCheckpointer.load_checkpoint("/tmp/mlxu/aba261c3aaa445e3b9347edaa6ee2dea/streaming_params_10241_0.37058815360069275")

params = {}

print("model loaded")
for k, v in tqdm(flatten_dict(model, sep=".").items()): # add ["params"]
    if "embed_tokens" in k:
        params[k.replace("embedding", "weight")] = torch.from_numpy(v.astype(np.float32)).to(dtype=torch.bfloat16)
    elif "kernel" in k:
        params[k.replace("kernel", "weight")] = torch.from_numpy(v.astype(np.float32)).to(dtype=torch.bfloat16).t().contiguous()
    else:
        params[k] = torch.from_numpy(v.astype(np.float32)).to(dtype=torch.bfloat16)

save_file(params, "gemma-2b-013/model.safetensors", {"format": "pt"})