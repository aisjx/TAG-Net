export CUDA_VISIBLE_DEVICES=0,1,2,3

# APAVA Dataset
# Explainability visualizations:
# 1) dual-view frequency/temporal contribution to classification decisions
# 2) frequency-temporal branch similarity
# 3) frequency/temporal graph adjacency heatmaps
# 4) multi-resolution channel-token similarity heatmaps
# 5) resolution-to-resolution interaction maps

ROOT_PATH=/home/ljh2025/ljh/DMKformer0/dataset/APAVA
TASK_NAME=classification
MODEL=TAGNet
DATA=APAVA

MODEL_ID_FULL=APAVA-Subject
MODEL_ID_CHANRES=APAVA-Subject-AblChanRes

E_LAYERS=4
BATCH_SIZE=64
VIS_BATCH_SIZE=32
D_MODEL=256
D_FF=512
N_HEADS=8
RES_LIST=4,6,8,16
NODEDIM=10
AUGS=none
TRAIN_AUGS=none,drop0.35
LR=0.0003
SEED=41
GPU_ID=0
MAX_SAMPLES=300
SPLIT=TEST

FULL_TAG=base
CHANRES_TAG=noChanResEnc

FULL_SETTING=${MODEL_ID_FULL}_${MODEL}_${DATA}_dm${D_MODEL}_df${D_FF}_nh${N_HEADS}_el${E_LAYERS}_res${RES_LIST}_node${NODEDIM}_${FULL_TAG}_seed${SEED}_bs${BATCH_SIZE}_lr${LR}
CHANRES_SETTING=${MODEL_ID_CHANRES}_${MODEL}_${DATA}_dm${D_MODEL}_df${D_FF}_nh${N_HEADS}_el${E_LAYERS}_res${RES_LIST}_node${NODEDIM}_${CHANRES_TAG}_seed${SEED}_bs${BATCH_SIZE}_lr${LR}

FULL_CKPT=./checkpoints/${TASK_NAME}/${MODEL_ID_FULL}/${MODEL}/${FULL_SETTING}/checkpoint.pth
CHANRES_CKPT=./checkpoints/${TASK_NAME}/${MODEL_ID_CHANRES}/${MODEL}/${CHANRES_SETTING}/checkpoint.pth

echo "Full checkpoint: ${FULL_CKPT}"
echo "Channel-resolution ablation checkpoint: ${CHANRES_CKPT}"

echo "=== Decision contribution: frequency view vs temporal view ==="
python \
  -u visualize_decision_view_contribution.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --checkpoint ${FULL_CKPT} \
  --split ${SPLIT} \
  --max_samples ${MAX_SAMPLES} \
  --batch_size ${BATCH_SIZE} \
  --num_workers 0 \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations ${AUGS} \
  --output_dir ./visualizations/APAVA_view_contribution \
  --output_name APAVA_TEST_dual_view_decision_contribution \
  --gpu ${GPU_ID}

echo "=== Cross-branch similarity: Full vs w/o Channel-Resolution ==="
python \
  -u visualize_full_vs_ablation_branch_similarity.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${CHANRES_CKPT} \
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
  --augmentations ${AUGS} \
  --output_dir ./visualizations/APAVA_branch_similarity \
  --output_name APAVA_TEST_branch_similarity_full_vs_noChanRes \
  --gpu ${GPU_ID}

echo "=== Average learned adjacency heatmaps: Full vs w/o Channel-Resolution ==="
python \
  -u visualize_full_vs_ablation_adjacency_heatmap.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${CHANRES_CKPT} \
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
  --augmentations ${AUGS} \
  --output_dir ./visualizations/APAVA_adjacency_heatmap \
  --output_name APAVA_TEST_avg_adjacency_full_vs_noChanRes \
  --gpu ${GPU_ID}

echo "=== Average token channel similarity heatmaps: Full vs w/o Channel-Resolution ==="
python \
  -u visualize_full_vs_ablation_token_heatmap.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${CHANRES_CKPT} \
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
  --augmentations ${AUGS} \
  --output_dir ./visualizations/APAVA_token_heatmap \
  --output_name APAVA_TEST_channel_centered_token_heatmap_full_vs_noChanRes \
  --gpu ${GPU_ID}

echo "=== Resolution-to-resolution interaction maps: Full vs w/o Channel-Resolution ==="
python \
  -u visualize_resolution_interaction_map.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --ablation_checkpoint ${CHANRES_CKPT} \
  --ablation_type no_channel_resolution \
  --split ${SPLIT} \
  --branch both \
  --max_samples ${MAX_SAMPLES} \
  --batch_size ${VIS_BATCH_SIZE} \
  --num_workers 0 \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations ${AUGS} \
  --output_dir ./visualizations/APAVA_resolution_interaction \
  --output_name APAVA_TEST_resolution_interaction_full_vs_noChanRes \
  --gpu ${GPU_ID}
