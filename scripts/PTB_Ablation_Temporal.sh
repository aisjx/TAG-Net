export CUDA_VISIBLE_DEVICES=0,1,2,3

# PTB Dataset
# Ablation: w/o Temporal Branch
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
  --model_id PTB-AblTemp \
  --model TAGNet \
  --data PTB \
  --e_layers 6 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 8 \
  --resolution_list 2,4,8,16 \
  --nodedim 8 \
  --augmentations drop0.2 \
  --disable_temporal_branch \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.0003 \
  --train_epochs 15 \
  --patience 8 \
  --gpu 3