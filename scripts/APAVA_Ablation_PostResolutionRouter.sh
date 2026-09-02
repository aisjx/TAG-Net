export CUDA_VISIBLE_DEVICES=0,1,2,3

# APAVA Dataset
# Ablation: w/o Post-Resolution Router
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/ljh2025/ljh/DMKformer0/dataset/APAVA \
  --model_id APAVA-Subject-AblPostRouter \
  --model TAGNet \
  --data APAVA \
  --e_layers 4 \
  --batch_size 64 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --resolution_list 4,6,8,16 \
  --nodedim 10 \
  --augmentations none,drop0.35 \
  --disable_post_resolution_router \
  --des 'Exp' \
  --itr 15 \
  --learning_rate 0.0003 \
  --train_epochs 100 \
  --patience 10 \
  --gpu 0
