export CUDA_VISIBLE_DEVICES=0,1,2,3
# PTB Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB/ \
  --model_id PTB \
  --model TAGNet \
  --data PTB \
  --e_layers 6 \
  --batch_size 64 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --resolution_list 2,4,6,8,16 \
  --nodedim 8 \
  --augmentations jitter0.2,drop0.2 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0003 \
  --train_epochs 100 \
  --patience 10 \
  --gpu 3