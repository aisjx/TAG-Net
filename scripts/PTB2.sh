export CUDA_VISIBLE_DEVICES=0,1,2,3
# PTB Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB/ \
  --model_id PTB \
  --model MedGNN \
  --data PTB \
  --e_layers 2 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 8 \
  --resolution_list 4,8 \
  --nodedim 8 \
  --augmentations jitter0.1,scale0.15,mask0.05 \
  --dropout 0.3 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0003 \
  --train_epochs 100 \
  --patience 10 \
  --gpu 2