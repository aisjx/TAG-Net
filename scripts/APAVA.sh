export CUDA_VISIBLE_DEVICES=0,1,2,3

ROOT_PATH=/home/ljh2025/ljh/DMKformer0/dataset/APAVA
TASK_NAME=classification
MODEL=MedGNN
DATA=APAVA

MODEL_ID_FULL=APAVA-Subject
MODEL_ID_ABL=APAVA-Subject-AblChanRes

E_LAYERS=4
BATCH_SIZE=64
D_MODEL=256
D_FF=512
N_HEADS=8
RES_LIST=4,6,8,16
NODEDIM=10
AUGS=none,drop0.35
ITR=1
LR=0.0003
EPOCHS=1
PATIENCE=10
GPU_ID=0
SEED=41

FULL_TAG=base
ABL_TAG=noChanResEnc

FULL_SETTING=${MODEL_ID_FULL}_${MODEL}_${DATA}_dm${D_MODEL}_df${D_FF}_nh${N_HEADS}_el${E_LAYERS}_res${RES_LIST}_node${NODEDIM}_${FULL_TAG}_seed${SEED}_bs${BATCH_SIZE}_lr${LR}
ABL_SETTING=${MODEL_ID_ABL}_${MODEL}_${DATA}_dm${D_MODEL}_df${D_FF}_nh${N_HEADS}_el${E_LAYERS}_res${RES_LIST}_node${NODEDIM}_${ABL_TAG}_seed${SEED}_bs${BATCH_SIZE}_lr${LR}

FULL_CKPT=./checkpoints/${TASK_NAME}/${MODEL_ID_FULL}/${MODEL}/${FULL_SETTING}/checkpoint.pth
ABL_CKPT=./checkpoints/${TASK_NAME}/${MODEL_ID_ABL}/${MODEL}/${ABL_SETTING}/checkpoint.pth

echo "Full checkpoint: ${FULL_CKPT}"
echo "Ablation checkpoint: ${ABL_CKPT}"

echo "=== Train Full Model ==="
python \
  -u run.py \
  --task_name ${TASK_NAME} \
  --is_training 1 \
  --root_path ${ROOT_PATH} \
  --model_id ${MODEL_ID_FULL} \
  --model ${MODEL} \
  --data ${DATA} \
  --e_layers ${E_LAYERS} \
  --batch_size ${BATCH_SIZE} \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations ${AUGS} \
  --des 'Exp' \
  --itr ${ITR} \
  --learning_rate ${LR} \
  --train_epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --gpu ${GPU_ID}

echo "=== Train w/o Channel-Resolution Encoder ==="
python \
  -u run.py \
  --task_name ${TASK_NAME} \
  --is_training 1 \
  --root_path ${ROOT_PATH} \
  --model_id ${MODEL_ID_ABL} \
  --model ${MODEL} \
  --data ${DATA} \
  --e_layers ${E_LAYERS} \
  --batch_size ${BATCH_SIZE} \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations ${AUGS} \
  --disable_channel_resolution_module \
  --des 'Exp' \
  --itr ${ITR} \
  --learning_rate ${LR} \
  --train_epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --gpu ${GPU_ID}

echo "=== Visualize Final Representation: Full vs w/o Channel-Resolution Encoder ==="
python \
  -u visualize_full_vs_ablation_tsne.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${ABL_CKPT} \
  --split TEST \
  --method tsne \
  --max_samples 300 \
  --batch_size ${BATCH_SIZE} \
  --num_workers 0 \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations none \
  --output_dir ./visualizations/APAVA_full_vs_chanres \
  --output_name APAVA_TEST_tsne_joint_full_vs_noChanRes \
  --gpu ${GPU_ID}

echo "=== Visualize Cross-Branch Similarity: Full vs w/o Channel-Resolution Encoder ==="
python \
  -u visualize_full_vs_ablation_branch_similarity.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${ABL_CKPT} \
  --split TEST \
  --max_samples 300 \
  --batch_size ${BATCH_SIZE} \
  --num_workers 0 \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations none \
  --output_dir ./visualizations/APAVA_branch_similarity \
  --output_name APAVA_TEST_branch_similarity_full_vs_noChanRes \
  --gpu ${GPU_ID}

echo "=== Visualize Average Token Channel Correlation Heatmaps: Full vs w/o Channel-Resolution Encoder ==="
python \
  -u visualize_full_vs_ablation_token_heatmap.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${ABL_CKPT} \
  --split TEST \
  --branch both \
  --metric cosine \
  --max_samples 300 \
  --batch_size 32 \
  --num_workers 0 \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations none \
  --output_dir ./visualizations/APAVA_token_heatmap \
  --output_name APAVA_TEST_avg_token_heatmap_full_vs_noChanRes \
  --gpu ${GPU_ID}

echo "=== Visualize Average Learned Adjacency Heatmaps: Full vs w/o Channel-Resolution Encoder ==="
python \
  -u visualize_full_vs_ablation_adjacency_heatmap.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${ABL_CKPT} \
  --split TEST \
  --branch both \
  --max_samples 300 \
  --batch_size 32 \
  --num_workers 0 \
  --edge_threshold 0.05 \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations none \
  --output_dir ./visualizations/APAVA_adjacency_heatmap \
  --output_name APAVA_TEST_avg_adjacency_full_vs_noChanRes \
  --gpu ${GPU_ID}