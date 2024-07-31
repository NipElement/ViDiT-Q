EXP_NAME='w4a8_timestep_aware_cb'

CFG="./t2v/configs/quant/opensora/16x512x512.py" # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # splited ckpt generated by split_ckpt.py
OUTDIR="./logs/$EXP_NAME"  # the path of the result of the W4A8 PTQ
GPU_ID=$1
MP_W_CONFIG="./t2v/configs/quant/opensora/mixed_precision/t20_weight_4_mp.yaml"  # the mixed precision config of weight
MP_A_CONFIG="./t2v/configs/quant/opensora/mixed_precision/t20_act_8_mp.yaml" # the mixed precision config of act
#SAVE_DIR="W4A8_Naive_Smooth_samples"  # leave blank to use the default path $OUTDIR/generated_videos

# quant infer
CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/quant_txt2video_mp.py $CFG --outdir $OUTDIR --ckpt_path $CKPT_PATH  --dataset_type opensora \
	--part_fp\
	--timestep_wise_mp \
	--time_mp_config_weight $MP_W_CONFIG \
	--time_mp_config_act $MP_A_CONFIG \
	--precompute_text_embeds ./t2v/utils_files/text_embeds.pth \
	#--save_dir $SAVE_DIR
