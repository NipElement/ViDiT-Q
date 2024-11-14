import os
import sys
sys.path.append("/home/yuansheng/TIGER-Lab/DiT/VG/Allegro")

import torch
import torch.nn as nn
import logging
from tqdm import trange
import numpy as np

from transformers import T5EncoderModel, T5Tokenizer
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel

from qdiff.models.quant_model import QuantModel
from qdiff.models.quant_layer_pixart import QuantLayer

logger = logging.getLogger(__name__)

class QuantConfig:
    def __init__(self, config_dict):
        self._config = config_dict
        for key, value in config_dict.items():
            setattr(self, key, value)
            
    def get(self, key, default=None):
        return self._config.get(key, default)

class AllegroQuantModel(QuantModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'allegro'
        
    def quant_layer_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, prefix=""):
        """重写quant_layer_refactor方法，只量化Linear层"""
        for name, child_module in module.named_children():
            full_name = prefix + name
            if isinstance(child_module, nn.Linear):  # 只量化Linear层
                setattr(
                    module,
                    name,
                    QuantLayer(
                        child_module,
                        weight_quant_params,
                        act_quant_params
                    )
                )
            else:
                self.quant_layer_refactor(child_module, weight_quant_params, act_quant_params, prefix=full_name+'.')

def load_fp_layers(fp_list_path):
    if not os.path.exists(fp_list_path):
        return []
    with open(fp_list_path, 'r') as f:
        fp_layers = [line.strip() for line in f.readlines()]
    return fp_layers

def main():
    # 1. 配置
    cfg = {
        "model": {
            "config_path": "/data/yuansheng/model/Allegro/transformer/config.json",
            "weights_path": "/data/yuansheng/model/DiT-Allegro/Allegro-v1-split-qkv.pth",
            "space_scale": 0.5,
            "time_scale": 1.0
        }
    }
    
    # 加载需要保持FP16的层
    fp_layer_list = load_fp_layers("./t2v/configs/quant/allegro/remain_fp.txt")
    
    device = torch.device('cuda')
    
    text_encoder = T5EncoderModel.from_pretrained(
        "/data/yuansheng/model/Allegro/text_encoder",
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    # 2. 构建模型
    transformer = AllegroTransformer3DModel.from_config(
        cfg['model']['config_path'],
        space_scale=cfg['model']['space_scale'],
        time_scale=cfg['model']['time_scale']
    )
    state_dict = torch.load(cfg['model']['weights_path'], map_location=device)
    transformer.load_state_dict(state_dict)
    transformer = transformer.to(device, dtype=torch.bfloat16).eval()
    
    # 3. 量化参数
    weight_quant_params = QuantConfig({
        "n_bits": 8,
        "channel_wise": True,
        "per_group": "channel",
        "channel_dim": 0,
        "scale_method": "min_max",
        "round_mode": "nearest"
    })
    
    act_quant_params = QuantConfig({
        "n_bits": 8,
        "per_group": False,
        "scale_method": "min_max",
        "round_mode": "nearest_ste",
        "running_stat": False,
        "dynamic": False,
        "sym": False,
        "n_spatial_token": 921600,  # for 88x720x1280
        "n_temporal_token": 88,
        "n_prompt": 512
    })
    
    # 4. 构建量化模型
    qnn = AllegroQuantModel(
        model=transformer,
        weight_quant_params=weight_quant_params,
        act_quant_params=act_quant_params,
    )
    qnn.cuda()
    qnn.eval()
    
    # 设置模型属性
    qnn.cfg_split = False
    qnn.set_grad_ckpt(False)
    
    # 设置量化器名称
    qnn.set_module_name_for_quantizer(module=qnn.model)
    
    # 5. 加载校准数据
    calib_data_dict = torch.load("/home/yuansheng/TIGER-Lab/DiT/outputs/calib_data/calib_data.pt")
    calib_xs = calib_data_dict["xs"].to(torch.bfloat16)  # 视频潜在表示
    calib_ts = calib_data_dict["ts"].to(torch.bfloat16)  # 时间步长
    calib_cs = calib_data_dict["cs"]  # 文本条件

    # 使用文本编码器处理文本输入
    with torch.no_grad():
        calib_cs = text_encoder(
            calib_cs,
            output_hidden_states=True,
        ).last_hidden_state.to(torch.bfloat16)

    print("处理后的校准数据维度:")
    print(f"xs shape: {calib_xs.shape}")
    print(f"ts shape: {calib_ts.shape}")
    print(f"cs shape: {calib_cs.shape}")

    # 调整cs的维度为 [B, 1, L, D]
    if len(calib_cs.shape) == 3:
        # [B, L, D] -> [B, 1, L, D]
        calib_cs = calib_cs.unsqueeze(1)

    print("调整后的cs维度:", calib_cs.shape)

    calib_batch_size = 1
    
    # 创建attention mask和encoder attention mask
    batch_size = calib_xs.shape[0]
    attention_mask = torch.ones(
        (batch_size, calib_xs.shape[2], calib_xs.shape[3], calib_xs.shape[4]), 
        dtype=torch.bool, 
        device=device
    )
    encoder_attention_mask = torch.ones(
        (batch_size, 1, calib_cs.shape[2]),  # [B, 1, L]
        dtype=torch.bool,
        device=device
    )
    
    # 6. 量化过程
    with torch.no_grad():
        # 1. 权重量化
        qnn.set_quant_state(True, False)  # 只开启权重量化
        if fp_layer_list:
            qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, 
                              quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
        
        # 初始化权重量化
        _ = qnn(
            calib_xs[:calib_batch_size].cuda(),
            calib_ts[:calib_batch_size].cuda(),
            calib_cs[:calib_batch_size].cuda(),
            attention_mask=attention_mask[:calib_batch_size].cuda(),
            encoder_attention_mask=encoder_attention_mask[:calib_batch_size].cuda()
        )
        qnn.set_quant_init_done('weight')
        
        # 2. 激活量化
        qnn.set_quant_state(True, True)  # 同时开启权重和激活量化
        if fp_layer_list:
            qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, 
                              quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
        
        # 分批处理校准数据
        inds = np.arange(calib_xs.shape[0])
        rounds = int(calib_xs.size(0) / calib_batch_size)
        
        for i in trange(rounds):
            batch_slice = slice(i * calib_batch_size, (i + 1) * calib_batch_size)
            _ = qnn(
                calib_xs[batch_slice].cuda(),
                calib_ts[batch_slice].cuda(),
                calib_cs[batch_slice].cuda(),
                attention_mask=attention_mask[batch_slice].cuda(),
                encoder_attention_mask=encoder_attention_mask[batch_slice].cuda()
            )
        qnn.set_quant_init_done('activation')
    
    # 7. 保存量化参数
    os.makedirs("./outputs", exist_ok=True)
    quant_params_dict = qnn.get_quant_params_dict()
    torch.save(quant_params_dict, "./outputs/w8a8_naive.pth")

if __name__ == "__main__":
    main()