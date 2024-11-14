import os
import sys
sys.path.append("/home/yuansheng/TIGER-Lab/DiT/VG/Allegro")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel
from allegro.pipelines.pipeline_allegro import AllegroPipeline

from qdiff.models.quant_model import QuantModel
from qdiff.models.quant_layer_pixart import QuantLayer
from qdiff.utils import load_quant_params
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

    def forward(self, *args, **kwargs):
        """重写forward方法，直接传递所有参数给底层模型"""
        return self.model(*args, **kwargs)


def single_inference(args):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16

    # 1. Load base models
    vae = AllegroAutoencoderKL3D.from_pretrained(
        args.vae, 
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder, 
        torch_dtype=dtype
    ).to(device)
    text_encoder.eval()

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)

    # 2. Build transformer model
    transformer = AllegroTransformer3DModel.from_config(
        args.dit,
        space_scale=0.5,
        time_scale=1.0
    ).to(device)
    
    # 3. Initialize QuantModel with fixed parameters
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
    qnn = AllegroQuantModel(
        model=transformer,
        weight_quant_params=weight_quant_params,
        act_quant_params=act_quant_params
    ).to(device)
    qnn.eval()

    # 4. Set quantization state
    qnn.set_quant_state(False, False)  # Enable weight and activation quantization
    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')

    # 5. Load quantized parameters
    load_quant_params(qnn, args.quant_ckpt, dtype=dtype)
    qnn.to(device)
    qnn.to(dtype)

    # 6. Build pipeline with quantized model
    scheduler = EulerAncestralDiscreteScheduler()

    allegro_pipeline = AllegroPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=qnn,
        scheduler=scheduler
    ).to(device)
    
#     positive_prompt = """
# (masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
# {} 
# emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
# sharp focus, high budget, cinemascope, moody, epic, gorgeous
# """

#     negative_prompt = """
# nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
# low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
# """

#     user_prompt = positive_prompt.format(args.user_prompt.lower().strip())

    # 7. Generate video
    sample = allegro_pipeline(
        prompt=args.user_prompt,
        # negative_prompt = negative_prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_sampling_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=device).manual_seed(args.seed)
    )
    
    print(f"Type of result: {type(sample)}")
    print(f"video shape:", sample.shape if torch.is_tensor(sample) else "no shape")            
    out_video=sample[0].video
    if isinstance(out_video, torch.Tensor):
        out_video = out_video.cpu().numpy()
        
        out_video = out_video.squeeze(0)
        
        if out_video.max() <= 1.0:
            out_video = (out_video * 255).astype(np.uint8)
        else:
            out_video = out_video.astype(np.uint8)
        if out_video.shape[-1] != 3:
            out_video = np.transpose(out_video, (0, 2, 3, 1))
            
    print(f"Final video shape: {out_video.shape}")

    # 8. Save video
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    import imageio
    imageio.mimwrite(args.save_path, out_video, fps=15, quality=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_prompt", type=str, default="A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this location might be a popular spot for docking fishing boats.")
    parser.add_argument("--vae", type=str, default="/data/yuansheng/model/Allegro/vae")
    parser.add_argument("--dit", type=str, default="/data/yuansheng/model/Allegro/transformer/config.json")
    parser.add_argument("--text_encoder", type=str, default="/data/yuansheng/model/Allegro/text_encoder")
    parser.add_argument("--tokenizer", type=str, default="/data/yuansheng/model/Allegro/tokenizer")
    parser.add_argument("--quant_ckpt", type=str, default="./outputs/w8a8_naive.pth")
    parser.add_argument("--save_path", type=str, default="./outputs/test_ptq/test_video.mp4")
    parser.add_argument("--num_frames", type=int, default=88)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_sampling_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    single_inference(args)