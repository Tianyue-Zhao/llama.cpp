from safetensors import safe_open
from gguf import *
import os
import torch

model_directory = "../glm-4v-9b"
# Gather the part files
model_files = os.listdir(model_directory)
model_parts = [
    model_file for model_file in model_files if\
    model_file.startswith("model-") and model_file.endswith(".safetensors")
]

vision_prefix = "transformer.vision."

output_file = "glm-clip.gguf"

# Load the model part files
model_tensors = {}
for model_part in model_parts:
    with safe_open(os.path.join(model_directory, model_part), framework="pt", device="cpu") as input:
        for key in input.keys():
            if not key.startswith(vision_prefix):
                continue
            if "cls_embedding" in key:
                print("Skipping class embedding")
                continue
            model_tensors[key] = input.get_tensor(key)

# Tensor name map for tensors in each layer
layer_map = {
    "attention.dense": "attn_out",
    "attention.query_key_value": "attn_qkv",
    "input_layernorm": "ln1",
    "post_attention_layernorm": "ln2",
    "mlp.fc1": "ffn_down",
    "mlp.fc2": "ffn_up"
}
layer_prefix_torch = "transformer.layers"
layer_prefix_gguf = "v.blk"
qkv_tensor_name = "attn_qkv"
q_tensor_name = "attn_q"
k_tensor_name = "attn_k"
v_tensor_name = "attn_v"

# Tensor name map for tensors not in the transformer
tensor_map = {
    "patch_embedding.proj.weight": "v.patch_embd.weight",
    "patch_embedding.proj.bias": "v.patch_embd.bias",
    "patch_embedding.position_embedding": "v.position_embd",
    "conv": "adapter.conv",
    "linear_proj.linear_proj": "adapter.linear.linear",
    "linear_proj.norm1": "adapter.linear.norm1",
    "linear_proj.dense_h_to_4h": "adapter.linear.dense_h_to_4h",
    "linear_proj.gate_proj": "adapter.linear.gate",
    "linear_proj.dense_4h_to_h": "adapter.linear.dense_4h_to_h",
    "boi": "adapter.boi",
    "eoi": "adapter.eoi"
}

# Map tensor names
key_snapshot = list(model_tensors.keys())
unmapped_tensors = set()
for key in key_snapshot:
    # Look for layer prefix
    edit_key = key.replace(vision_prefix, '')
    if layer_prefix_torch in key:
        edit_key = edit_key.replace(layer_prefix_torch, layer_prefix_gguf)
        flag = False
        # Loop through the layer tensor names
        for layer_key in layer_map:
            if layer_key in key:
                edit_key = edit_key.replace(layer_key, layer_map[layer_key])
                print(f"Mapped tensor {key} to {edit_key}")
                model_tensors[edit_key] = model_tensors[key]
                del model_tensors[key]
                flag = True
                break
        if not flag:
            unmapped_tensors.add(key)
            del model_tensors[key]
        # Split the QKV tensor into three tensors
        if qkv_tensor_name in edit_key:
            single_size = model_tensors[edit_key].shape[0] // 3
            model_tensors[edit_key.replace(qkv_tensor_name, q_tensor_name)] =\
                model_tensors[edit_key][:single_size, ...]
            model_tensors[edit_key.replace(qkv_tensor_name, k_tensor_name)] =\
                model_tensors[edit_key][single_size : 2 * single_size, ...]
            model_tensors[edit_key.replace(qkv_tensor_name, v_tensor_name)] =\
                model_tensors[edit_key][2 * single_size:, ...]
            del model_tensors[edit_key]
    else:
        # Loop through tensor names
        for global_key in tensor_map:
            flag = False
            if global_key in key:
                edit_key = edit_key.replace(global_key, tensor_map[global_key])
                print(f"Mapped tensor {key} to {edit_key}")
                model_tensors[edit_key] = model_tensors[key]
                del model_tensors[key]
                flag = True
                break
        if not flag:
            unmapped_tensors.add(key)
            del model_tensors[key]

if len(unmapped_tensors) > 0:
    print("Warning: Some tensors in the checkpoint were not utilized")
    print('\n'.join(list(unmapped_tensors)))

default_image_mean = [0.5, 0.5, 0.5]
default_image_std = [0.5, 0.5, 0.5]

def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)

fout = GGUFWriter(path=os.path.join(model_directory, output_file), arch="clip")
fout.add_name("glm-4v-9b")
fout.add_bool("clip.has_text_encoder", False)
fout.add_bool("clip.has_vision_encoder", True)
fout.add_bool("clip.has_glm_projector", True)
fout.add_file_type(0)  # float 32
fout.add_string("clip.projector_type", "glm4v")
fout.add_uint32("clip.vision.image_size", 1120)
fout.add_uint32("clip.vision.patch_size", 14)
fout.add_uint32(k(KEY_EMBEDDING_LENGTH, "clip.vision"), 1792)
fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, "clip.vision"), 15360)
fout.add_uint32("clip.vision.projection_dim", 0)
fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, "clip.vision"), 16)
fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, "clip.vision"), 1e-6)
fout.add_uint32(k(KEY_BLOCK_COUNT, "clip.vision"), 63)
fout.add_array("clip.vision.image_mean", default_image_mean)
fout.add_array("clip.vision.image_std", default_image_std)
fout.add_bool("clip.use_gelu", True)

for key in model_tensors.keys():
    tensor = model_tensors[key]
    tensor = tensor.to(torch.float32).squeeze().numpy()
    fout.add_tensor(key, tensor)

fout.write_header_to_file()
fout.write_kv_data_to_file()
fout.write_tensors_to_file()
fout.close()