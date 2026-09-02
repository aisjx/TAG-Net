export CUDA_VISIBLE_DEVICES=0,1,2,3

# APAVA Dataset
# Explainability experiment:
# Compare dual-view graph learning with early-fusion single-GNN graph learning.
# Rows in the heatmap:
# 1) full frequency graph
# 2) full temporal graph
# 3) early-fusion single-GNN graph
# 4) absolute frequency-temporal graph difference

ROOT_PATH=/home/ljh2025/ljh/DMKformer0/dataset/APAVA
TASK_NAME=classification
MODEL=TAGNet
DATA=APAVA

MODEL_ID_FULL=APAVA-Subject
MODEL_ID_SINGLE=APAVA-Subject-SingleGNNAdd

E_LAYERS=4
BATCH_SIZE=64
VIS_BATCH_SIZE=32
D_MODEL=256
D_FF=512
N_HEADS=8
RES_LIST=4,6,8,16
NODEDIM=10
AUGS=none
LR=0.0003
SEED=41
GPU_ID=0
MAX_SAMPLES=300
SPLIT=TEST
EDGE_THRESHOLD=0.05
SINGLE_FUSION=add

FULL_TAG=base
SINGLE_TAG=singleGNN_sgFusionadd

FULL_SETTING=${MODEL_ID_FULL}_${MODEL}_${DATA}_dm${D_MODEL}_df${D_FF}_nh${N_HEADS}_el${E_LAYERS}_res${RES_LIST}_node${NODEDIM}_${FULL_TAG}_seed${SEED}_bs${BATCH_SIZE}_lr${LR}
SINGLE_SETTING=${MODEL_ID_SINGLE}_${MODEL}_${DATA}_dm${D_MODEL}_df${D_FF}_nh${N_HEADS}_el${E_LAYERS}_res${RES_LIST}_node${NODEDIM}_${SINGLE_TAG}_seed${SEED}_bs${BATCH_SIZE}_lr${LR}

FULL_CKPT=./checkpoints/${TASK_NAME}/${MODEL_ID_FULL}/${MODEL}/${FULL_SETTING}/checkpoint.pth
SINGLE_CKPT=./checkpoints/${TASK_NAME}/${MODEL_ID_SINGLE}/${MODEL}/${SINGLE_SETTING}/checkpoint.pth

echo "Full checkpoint: ${FULL_CKPT}"
echo "Single-GNN checkpoint: ${SINGLE_CKPT}"

echo "=== Dual-view graph vs early-fusion single-GNN adjacency visualization ==="
python \
  -u visualize_dual_vs_single_fused_adjacency.py \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --full_checkpoint ${FULL_CKPT} \
  --single_checkpoint ${SINGLE_CKPT} \
  --split ${SPLIT} \
  --max_samples ${MAX_SAMPLES} \
  --batch_size ${VIS_BATCH_SIZE} \
  --num_workers 0 \
  --edge_threshold ${EDGE_THRESHOLD} \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --n_heads ${N_HEADS} \
  --e_layers ${E_LAYERS} \
  --resolution_list ${RES_LIST} \
  --nodedim ${NODEDIM} \
  --augmentations ${AUGS} \
  --single_gnn_fusion ${SINGLE_FUSION} \
  --output_dir ./visualizations/APAVA_dual_vs_single_adjacency \
  --output_name APAVA_TEST_dual_vs_single_add_real_adjacency \
  --gpu ${GPU_ID}
