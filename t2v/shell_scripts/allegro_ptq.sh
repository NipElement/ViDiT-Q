EXP_NAME=${2:-"w8a8_naive"}

CFG="./t2v/configs/quant/allegro/88x720x1280.py"  # the allegro config
Q_CFG="./t2v/configs/quant/allegro/$EXP_NAME.yaml"  # TODO: the config of PTQ
CKPT_PATH="/data/yuansheng/model/DiT-Allegro/Allegro-v1-split-qkv.pth"  # splited ckpt generated by split_ckpt.py
CALIB_DATA_DIR="/home/yuansheng/TIGER-Lab/DiT/outputs/calib_data"  # your path of calib data
OUTDIR="/data/yuansheng/model/DiT-Allegro/quant/$EXP_NAME"  # TODO: your path to save the ptq result
GPU_ID=$1

# ptq
CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/allegro_ptq.py \
    --config $CFG \
    --ckpt_path $CKPT_PATH \
    --ptq_config $Q_CFG \
    --outdir $OUTDIR \
    --calib_data $CALIB_DATA_DIR/calib_data.pt \
    --part_fp