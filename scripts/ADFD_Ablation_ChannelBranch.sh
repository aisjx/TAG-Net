export CUDA_VISIBLE_DEVICES=0,1,2,3

# ADFD Dataset
# Ablation: w/o Channel Branch (keep Resolution Router)
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
  --model_id ADFD-AblChanBranch \
  --model TAGNet \
  --data ADFD \
  --e_layers 6 \
  --batch_size 64 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --resolution_list 4,8,16 \
  --nodedim 8 \
  --augmentations none \
  --disable_channel_branch \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.0003 \
  --train_epochs 15 \
  --patience 8 \
  --gpu 0