import os
import sys
sys.path.append("/home/yuansheng/TIGER-Lab/DiT/VG/Allegro")

import torch
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from allegro.pipelines.pipeline_allegro import AllegroPipeline
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel
from mmengine.runner import set_random_seed

class CalibrationPipeline(AllegroPipeline):
    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            # 获取text embeddings
            text_embeddings = self.encode_prompt(
                args[0],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            
            # 准备latents
            latents = self.prepare_latents(
                batch_size=len(args[0]),
                num_channels_latents=4,
                height=kwargs.get("height", 720),
                width=kwargs.get("width", 1280),
                num_frames=kwargs.get("num_frames", 88),
                dtype=self.transformer.dtype,
                device=self.device,
                generator=None,
            )
            
            # 准备timesteps
            self.scheduler.set_timesteps(kwargs.get("num_inference_steps", 25), device=self.device)
            timesteps = self.scheduler.timesteps
            
            calib_data = {
                "xs": latents.detach(),
                "ts": timesteps.detach(),
                "cs": text_embeddings[1].detach()  # 只使用正向提示的embeddings
            }
            
            # 收集中间特征
            calib_features = {}
            def collect_features(module, input, output, name):
                if isinstance(output, tuple):
                    output = output[0]
                if any(key in name for key in [
                    'attn.output',
                    'mlp.output', 
                    'norm1.output',
                    'norm2.output'
                ]):
                    calib_features[name] = output.detach()
                    
            hooks = []
            for name, module in self.transformer.named_modules():
                if any(layer_type in name for layer_type in ['attn', 'mlp']):
                    hook = module.register_forward_hook(
                        lambda m, i, o, name=name: collect_features(m, i, o, name)
                    )
                    hooks.append(hook)

            samples = super().__call__(*args, **kwargs)
            
            for hook in hooks:
                hook.remove()

            return samples, calib_data, calib_features

def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts

def main():
    cfg = {
        "vae": {"from_pretrained": "/data/yuansheng/model/Allegro/vae"},
        "text_encoder": {"from_pretrained": "/data/yuansheng/model/Allegro/text_encoder"},
        "tokenizer": {"from_pretrained": "/data/yuansheng/model/Allegro/tokenizer"},
        "model": {
            "config_path": "/data/yuansheng/model/Allegro/transformer/config.json",
            "weights_path": "/data/yuansheng/model/DiT-Allegro/Allegro-v1-split-qkv.pth",
            "space_scale": 0.5,
            "time_scale": 1.0
        },
        "prompt_path": "/home/yuansheng/TIGER-Lab/DiT/ViDiT-Q/t2v/assets/texts/t2v_samples_simple_allegro.txt",
        "save_dir": "/home/yuansheng/TIGER-Lab/DiT/outputs/calib_data",
        "batch_size": 1,
        "num_frames": 88,
        "image_size": (720, 1280),
        "num_sampling_steps": 25,
        "guidance_scale": 7.5,
        "seed": 42,
        "gpu": 0,
        "fps": 15,
        "data_num": 3
    }

    device = f"cuda:{cfg['gpu']}"
    set_random_seed(cfg['seed'])
    prompts = load_prompts(cfg['prompt_path'])[:cfg['data_num']]

    vae = AllegroAutoencoderKL3D.from_pretrained(
        cfg['vae']['from_pretrained'],
        torch_dtype=torch.float32
    ).to(device).eval()
    
    text_encoder = T5EncoderModel.from_pretrained(
        cfg['text_encoder']['from_pretrained'],
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = T5Tokenizer.from_pretrained(cfg['tokenizer']['from_pretrained'])
    
    transformer = AllegroTransformer3DModel.from_config(
        cfg['model']['config_path'],
        space_scale=cfg['model']['space_scale'],
        time_scale=cfg['model']['time_scale']
    )
    state_dict = torch.load(cfg['model']['weights_path'], map_location=device)
    transformer.load_state_dict(state_dict)
    transformer = transformer.to(device, dtype=torch.bfloat16).eval()

    os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

    pipeline = CalibrationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=EulerAncestralDiscreteScheduler(),
        transformer=transformer
    ).to(device)

    os.makedirs(cfg['save_dir'], exist_ok=True)
    all_calib_features = {}
    input_data_list = []
    
    with torch.no_grad():
        for i in range(0, len(prompts), cfg['batch_size']):
            batch_prompts = prompts[i : i + cfg['batch_size']]
            
            samples, calib_data, calib_features = pipeline(
                batch_prompts,
                num_frames=cfg['num_frames'],
                height=cfg['image_size'][0],
                width=cfg['image_size'][1],
                num_inference_steps=cfg['num_sampling_steps'],
                guidance_scale=cfg['guidance_scale'],
            )

            # 合并特征
            for key, value in calib_features.items():
                if key not in all_calib_features:
                    all_calib_features[key] = []
                all_calib_features[key].append(value)
            
            input_data_list.append(calib_data)

            # 保存生成的样本
            print("samples type:", type(samples))
            print("samples shape:", samples.shape if torch.is_tensor(samples) else [type(s) for s in samples])
            for j, video_output in enumerate(samples[0].video):
                print(f"video {j} type:", type(video_output))
                print(f"video {j} shape:", video_output.shape if torch.is_tensor(video_output) else "no shape")
                save_path = os.path.join(cfg['save_dir'], f"samples_{i+j}.mp4")
                import imageio
                if torch.is_tensor(video_output):
                    video_output = video_output.cpu().numpy()
                if not isinstance(video_output, np.ndarray):
                    raise ValueError(f"Unexpected video type: {type(video_output)}")
                imageio.mimwrite(save_path, video_output, fps=cfg['fps'], quality=8)

            torch.cuda.empty_cache()

    processed_calib_data = {}
    for key in all_calib_features:
        processed_calib_data[key] = torch.cat(all_calib_features[key], dim=0)
    
    torch.save(processed_calib_data, os.path.join(cfg['save_dir'], "features.pt"))

    calib_data_dict = {
        "xs": torch.cat([data["xs"] for data in input_data_list], dim=0),
        "ts": torch.cat([data["ts"] for data in input_data_list], dim=0),
        "cs": torch.cat([data["cs"] for data in input_data_list], dim=0)
    }
    torch.save(calib_data_dict, os.path.join(cfg['save_dir'], "calib_data.pt"))

if __name__ == "__main__":
    main()