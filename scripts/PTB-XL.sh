export CUDA_VISIBLE_DEVICES=0,1,2,3

# PTB-XL Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path  /home/ljh2025/ljh/DMKformer0/dataset/PTB-XL/ \
  --model_id PTB-XL \
  --model MedGNN \
  --data PTB-XL \
  --e_layers 4 \
  --batch_size 64 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --resolution_list 4,6,8,10,12,14 \
  --nodedim 6 \
  --augmentations drop0.2 \
  --dropout 0.3 \
  --des 'Exp' \
  --itr 15 \
  --learning_rate 0.0003 \
  --train_epochs 15 \
  --patience 8 \
  --gpu 0