num_frames = 88
fps = 15
image_size = (720, 1280)

# Define model
model = dict(
    type="AllegroTransformer3DModel",
    space_scale=0.5,
    time_scale=1.0,
    enable_flashattn=False,
    enable_layernorm_kernel=False,
    from_pretrained="/data/yuansheng/model/DiT-Allegro/Allegro-v1-split-qkv.pth",
)
vae = dict(
    type="AllegroAutoencoderKL3D",
    from_pretrained="/data/yuansheng/model/Allegro/vae",
)
text_encoder = dict(
    type="t5",
    from_pretrained="/data/yuansheng/model/Allegro/text_encoder",
    save_pretrained="/data/yuansheng/model/DiT-Allegro/temp_files/checkpoints/text_encoder",
    model_max_length=512,
)
tokenizer = dict(
    type="T5Tokenizer",
    from_pretrained="/data/yuansheng/model/Allegro/tokenizer",
)
scheduler = dict(
    type="EulerAncestralDiscreteScheduler",
    num_sampling_steps=100,
    cfg_scale=7.5,
)
dtype = "fp16"

# Others
batch_size = 1
seed = 42
prompt_path = None
save_dir = "/home/yuansheng/TIGER-Lab/DiT/outputs/"
