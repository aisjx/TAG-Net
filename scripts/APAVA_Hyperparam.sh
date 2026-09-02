export CUDA_VISIBLE_DEVICES=0,1,2,3

# APAVA Dataset
# Hyperparameter sweeps: resolution_list, nodedim, structure_delta_scale

ROOT_PATH=/home/ljh2025/ljh/DMKformer0/dataset/APAVA
TASK_NAME=classification
MODEL=MedGNN
DATA=APAVA
BASE_MODEL_ID=APAVA-Subject

E_LAYERS=4
BATCH_SIZE=64
D_MODEL=256
D_FF=512
N_HEADS=8
BASE_RES_LIST=4,6,8,16
BASE_NODEDIM=10
BASE_STRUCTURE_DELTA_SCALE=0.1
AUGS=none,drop0.35
ITR=2
LR=0.0001
EPOCHS=100
PATIENCE=10
GPU_ID=0

run_exp() {
  local model_id=$1
  local res_list=$2
  local nodedim=$3
  local structure_delta_scale=$4

  echo "=== ${DATA} | ${model_id} | res=${res_list} | nodedim=${nodedim} | structure_delta_scale=${structure_delta_scale} ==="
  python \
    -u run.py \
    --task_name ${TASK_NAME} \
    --is_training 1 \
    --root_path ${ROOT_PATH} \
    --model_id ${model_id} \
    --model ${MODEL} \
    --data ${DATA} \
    --e_layers ${E_LAYERS} \
    --batch_size ${BATCH_SIZE} \
    --d_model ${D_MODEL} \
    --d_ff ${D_FF} \
    --n_heads ${N_HEADS} \
    --resolution_list ${res_list} \
    --nodedim ${nodedim} \
    --structure_delta_scale ${structure_delta_scale} \
    --augmentations ${AUGS} \
    --des 'Hyperparam' \
    --itr ${ITR} \
    --learning_rate ${LR} \
    --train_epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --gpu ${GPU_ID}
}

echo "=== Sweep 1: resolution_list / number of resolutions ==="
for res_list in 4 4,6 4,6,8 4,6,8,16 2,4,6,8,16; do
  safe_res=${res_list//,/x}
  run_exp "${BASE_MODEL_ID}-HP-Res${safe_res}" "${res_list}" "${BASE_NODEDIM}" "${BASE_STRUCTURE_DELTA_SCALE}"
done

echo "=== Sweep 2: nodedim / graph node dimension ==="
for nodedim in 4 8 10 12 16 32; do
  run_exp "${BASE_MODEL_ID}-HP-Node${nodedim}" "${BASE_RES_LIST}" "${nodedim}" "${BASE_STRUCTURE_DELTA_SCALE}"
done

echo "=== Sweep 3: structure_delta_scale / structure alignment strength ==="
for structure_delta_scale in 0 0.05 0.1 0.2 0.5; do
  safe_delta=${structure_delta_scale//./p}
  run_exp "${BASE_MODEL_ID}-HP-SDelta${safe_delta}" "${BASE_RES_LIST}" "${BASE_NODEDIM}" "${structure_delta_scale}"
done
