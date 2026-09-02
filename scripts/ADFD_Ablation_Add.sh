export CUDA_VISIBLE_DEVICES=0,1,2,3

# ADFD Dataset
# Supplementary experiment: single-GNN with pre-graph add fusion
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
  --model_id ADFD-SingleGNNAdd \
  --model MedGNN \
  --data ADFD \
  --e_layers 6 \
  --batch_size 64 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --resolution_list 4,8,16 \
  --nodedim 8 \
  --augmentations none \
  --single_gnn \
  --single_gnn_fusion add \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.0003 \
  --train_epochs 15 \
  --patience 8 \
  --gpu 0