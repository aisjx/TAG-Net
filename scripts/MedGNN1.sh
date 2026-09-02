# subject-dependen
# ADFD Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFD/ \
  --model_id ADFD-1 \
  --model MedGNN \
  --data ADFD \
  --e_layers 6 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 8 \
  --resolution_list 4,6,8 \
  --nodedim 8 \
  --augmentations none \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0002 \
  --train_epochs 10 \
  --patience 5

# subject-dependent
# ADFD Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFD/ \
  --model_id ADFD-2 \
  --model MedGNN \
  --data ADFD \
  --e_layers 6 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 8 \
  --resolution_list 4,6,8,10 \
  --nodedim 8 \
  --augmentations none \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0002 \
  --train_epochs 20 \
  --patience 10