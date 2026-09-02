export CUDA_VISIBLE_DEVICES=0,1,2,3

# TDBRAIN Dataset
# Ablation: w/o Channel-Aware Multi-Resolution Encoder
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/ljh2025/ljh/DMKformer0/dataset/TDBRAIN/ \
  --model_id TDBRAIN-AblChanRes \
  --model MedGNN \
  --data TDBRAIN \
  --e_layers 4 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 8 \
  --resolution_list 2,4,8 \
  --nodedim 8 \
  --augmentations drop0.25 \
  --disable_channel_resolution_module \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.0002 \
  --train_epochs 100 \
  --patience 10 \
  --gpu 2
