export CUDA_VISIBLE_DEVICES=0

# APAVA Dataset
# Deep Analysis: Compare Full Model vs w/o Channel-Resolution Encoder
# Focus on samples rescued by multi-scale interaction

ROOT_PATH=/home/ljh2025/ljh/DMKformer0/dataset/APAVA
DATA=APAVA
SPLIT=TEST

# Model checkpoints (update these paths after training)
FULL_CKPT="./checkpoints/APAVA-Subject_base/checkpoint.pth"
ABL_CKPT="./checkpoints/APAVA-Subject_noChanResEnc/checkpoint.pth"

python \
  -u analyze_ablation_mispredictions.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${ABL_CKPT} \
  --split ${SPLIT} \
  --model TAGNet \
  --e_layers 4 \
  --batch_size 64 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --resolution_list 4,6,8,16 \
  --nodedim 10 \
  --augmentations none,drop0.35 \
  --structure_delta_scale 0.1 \
  --output_dir ./analysis/APAVA_mispredictions \
  --gpu 0
