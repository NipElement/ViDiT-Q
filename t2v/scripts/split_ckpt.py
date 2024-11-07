import torch
from safetensors import safe_open
from safetensors.torch import save_file
def split_qkv(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'qkv' in key:
            prefix, suffix = key.split('.qkv.')
            q_key = prefix + '.q.' + suffix
            k_key = prefix + '.k.' + suffix
            v_key = prefix + '.v.' + suffix
            print(q_key,k_key,v_key)
            new_state_dict[q_key] = value[:value.size(0) // 3]
            new_state_dict[k_key] = value[value.size(0) // 3: 2 * (value.size(0) // 3)]
            new_state_dict[v_key] = value[2 * (value.size(0) // 3):]
        else:
            new_state_dict[key] = value
    return new_state_dict

# state_dict = torch.load('./logs/split_ckpt/OpenSora-v1-HQ-16x512x512.pth')  # your path of the original ckpt

model_path = '/data/yuansheng/model/Allegro/transformer/diffusion_pytorch_model.safetensors'
state_dict = {}
with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)

    
new_state_dict = split_qkv(state_dict)
# for pth
torch.save(new_state_dict, '/data/yuansheng/model/DiT-Allegro/Allegro-v1-split-qkv.pth')  # save the split ckpt

# for safetensors
# save_file(new_state_dict, '/data/yuansheng/model/DiT-Allegro/Allegro-v1-split-qkv.safetensors')

# torch.save(new_state_dict, './logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split-test.pth')  # split the qkv layer in the ckpt
