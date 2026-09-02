export CUDA_VISIBLE_DEVICES=0,1,2,3

ROOT_PATH=/home/ljh2025/ljh/DMKformer0/dataset/TDBRAIN/
TASK_NAME=classification
MODEL=MedGNN
DATA=TDBRAIN

MODEL_ID_FULL=TDBRAIN
MODEL_ID_ABL=TDBRAIN-AblChanRes

E_LAYERS=4
BATCH_SIZE=64
VIS_BATCH_SIZE=32
D_MODEL=128
D_FF=256
N_HEADS=8
RES_LIST=2,4,8
NODEDIM=8
TRAIN_AUGS=drop0.25
VIS_AUGS=none
ITR=1
LR=0.0002
EPOCHS=10
PATIENCE=10
GPU_ID=2
SEED=41
MAX_SAMPLES=300
SPLIT=TEST

FULL_TAG=base
ABL_TAG=noChanResEnc

FULL_SETTING=${MODEL_ID_FULL}_${MODEL}_${DATA}_dm${D_MODEL}_df${D_FF}_nh${N_HEADS}_el${E_LAYERS}_res${RES_LIST}_node${NODEDIM}_${FULL_TAG}_seed${SEED}_bs${BATCH_SIZE}_lr${LR}
ABL_SETTING=${MODEL_ID_ABL}_${MODEL}_${DATA}_dm${D_MODEL}_df${D_FF}_nh${N_HEADS}_el${E_LAYERS}_res${RES_LIST}_node${NODEDIM}_${ABL_TAG}_seed${SEED}_bs${BATCH_SIZE}_lr${LR}

FULL_CKPT=./checkpoints/${TASK_NAME}/${MODEL_ID_FULL}/${MODEL}/${FULL_SETTING}/checkpoint.pth
ABL_CKPT=./checkpoints/${TASK_NAME}/${MODEL_ID_ABL}/${MODEL}/${ABL_SETTING}/checkpoint.pth

echo "Full checkpoint: ${FULL_CKPT}"
echo "Ablation checkpoint: ${ABL_CKPT}"

echo "=== Train Full Model: TDBRAIN ==="
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
  --augmentations ${TRAIN_AUGS} \
  --des 'Exp' \
  --itr ${ITR} \
  --learning_rate ${LR} \
  --train_epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --gpu ${GPU_ID}

echo "=== Train w/o Channel-Resolution Encoder: TDBRAIN ==="
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
  --augmentations ${TRAIN_AUGS} \
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
  --split ${SPLIT} \
  --method tsne \
  --max_samples ${MAX_SAMPLES} \
  --batch_size ${BATCH_SIZE} \
  --num_workers 0 \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations ${VIS_AUGS} \
  --output_dir ./visualizations/TDBRAIN_full_vs_chanres \
  --output_name TDBRAIN_TEST_tsne_joint_full_vs_noChanRes \
  --gpu ${GPU_ID}

echo "=== Visualize Cross-Branch Similarity: Full vs w/o Channel-Resolution Encoder ==="
python \
  -u visualize_full_vs_ablation_branch_similarity.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${ABL_CKPT} \
  --split ${SPLIT} \
  --max_samples ${MAX_SAMPLES} \
  --batch_size ${BATCH_SIZE} \
  --num_workers 0 \
  --center_mode feature \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations ${VIS_AUGS} \
  --output_dir ./visualizations/TDBRAIN_branch_similarity \
  --output_name TDBRAIN_TEST_branch_similarity_full_vs_noChanRes \
  --gpu ${GPU_ID}

echo "=== Visualize Average Token Channel Correlation Heatmaps: Full vs w/o Channel-Resolution Encoder ==="
python \
  -u visualize_full_vs_ablation_token_heatmap.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${ABL_CKPT} \
  --split ${SPLIT} \
  --branch both \
  --metric cosine \
  --center_mode channel \
  --max_samples ${MAX_SAMPLES} \
  --batch_size ${VIS_BATCH_SIZE} \
  --num_workers 0 \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations ${VIS_AUGS} \
  --output_dir ./visualizations/TDBRAIN_token_heatmap \
  --output_name TDBRAIN_TEST_channel_centered_token_heatmap_full_vs_noChanRes \
  --gpu ${GPU_ID}

echo "=== Visualize Average Learned Adjacency Heatmaps: Full vs w/o Channel-Resolution Encoder ==="
python \
  -u visualize_full_vs_ablation_adjacency_heatmap.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${ABL_CKPT} \
  --split ${SPLIT} \
  --branch both \
  --max_samples ${MAX_SAMPLES} \
  --batch_size ${VIS_BATCH_SIZE} \
  --num_workers 0 \
  --edge_threshold 0.05 \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations ${VIS_AUGS} \
  --output_dir ./visualizations/TDBRAIN_adjacency_heatmap \
  --output_name TDBRAIN_TEST_avg_adjacency_full_vs_noChanRes \
  --gpu ${GPU_ID}
