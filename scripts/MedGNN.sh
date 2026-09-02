# APAVA Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/APAVA/ \
  --model_id APAVA-Subject \
  --model MedGNN \
  --data APAVA \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 8 \
  --resolution_list 2,4,6,8 \
  --nodedim 10 \
  --augmentations none,drop0.35 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --patience 8

# ADFD Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFD/ \
  --model_id ADFD \
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
  --train_epochs 15 \
  --patience 8


# PTB Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB/ \
  --model_id PTB \
  --model MedGNN \
  --data PTB \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 8 \
  --resolution_list 2,4,6,8 \
  --nodedim 10 \
  --augmentations drop0.2 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --patience 8

# TDBRAIN Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/TDBRAIN/ \
  --model_id TDBRAIN \
  --model MedGNN \
  --data TDBRAIN \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 8 \
  --resolution_list 2,4,6,8 \
  --nodedim 10 \
  --augmentations drop0.2 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --patience 8

# PTB-XL Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB-XL/ \
  --model_id PTB-XL \
  --model MedGNN \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 8 \
  --resolution_list 2,4,6,8 \
  --nodedim 10 \
  --augmentations drop0.2 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --patience 8