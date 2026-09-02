#!/bin/bash

# ==========================================
# ADFD Dataset - Run All Ablations (Parallel)
# Part 1 (Branch)      → GPU 0
# Part 2 (ChannelRes)  → GPU 2
# Part 3 (DualGraph)   → GPU 3
# ==========================================

echo "=========================================="
echo "Starting ADFD Ablation Experiments (Parallel)"
echo "Part 1 (Branch)     → GPU 0"
echo "Part 2 (ChannelRes) → GPU 2"
echo "Part 3 (DualGraph)  → GPU 3"
echo "=========================================="

# ==================== Part 1: Branch Ablation (GPU 0) ====================
run_part1() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[Part 1] Started on GPU 0"

    echo "  [1/10] w/o Temporal Branch..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
        --model_id ADFD-AblTemp --model TAGNet --data ADFD \
        --e_layers 6 --batch_size 64 --d_model 256 --d_ff 512 --n_heads 8 \
        --resolution_list 4,8,16 --nodedim 8 --augmentations none \
        --disable_temporal_branch --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [1/10]"

    echo "  [2/10] w/o Frequency Branch..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
        --model_id ADFD-AblFreq --model TAGNet --data ADFD \
        --e_layers 6 --batch_size 64 --d_model 256 --d_ff 512 --n_heads 8 \
        --resolution_list 4,8,16 --nodedim 8 --augmentations none \
        --disable_frequency_branch --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [2/10]"

    echo "[Part 1] All experiments completed on GPU 0"
}

# ==================== Part 2: Channel-Aware Resolution Encoder Ablation (GPU 2) ====================
run_part2() {
    export CUDA_VISIBLE_DEVICES=2
    echo "[Part 2] Started on GPU 2"

    echo "  [3/10] w/o Channel Branch..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
        --model_id ADFD-AblChanBranch --model TAGNet --data ADFD \
        --e_layers 6 --batch_size 64 --d_model 256 --d_ff 512 --n_heads 8 \
        --resolution_list 4,8,16 --nodedim 8 --augmentations none \
        --disable_channel_branch --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [3/10]"

    echo "  [4/10] w/o Resolution Router..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
        --model_id ADFD-AblResRouter --model TAGNet --data ADFD \
        --e_layers 6 --batch_size 64 --d_model 256 --d_ff 512 --n_heads 8 \
        --resolution_list 4,8,16 --nodedim 8 --augmentations none \
        --disable_resolution_router --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [4/10]"

    echo "  [5/10] w/o Channel-Aware Multi-Resolution Encoder..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
        --model_id ADFD-AblChanRes --model TAGNet --data ADFD \
        --e_layers 6 --batch_size 64 --d_model 256 --d_ff 512 --n_heads 8 \
        --resolution_list 4,8,16 --nodedim 8 --augmentations none \
        --disable_channel_resolution_module --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [5/10]"

    echo "[Part 2] All experiments completed on GPU 2"
}

# ==================== Part 3: Dual-Graph Interaction Ablation (GPU 3) ====================
run_part3() {
    export CUDA_VISIBLE_DEVICES=3
    echo "[Part 3] Started on GPU 3"

    echo "  [6/10] w/o Dual-Graph Interaction..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
        --model_id ADFD-AblDualGraph --model TAGNet --data ADFD \
        --e_layers 6 --batch_size 64 --d_model 256 --d_ff 512 --n_heads 8 \
        --resolution_list 4,8,16 --nodedim 8 --augmentations none \
        --disable_cross_graph_interaction --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [6/10]"

    echo "  [7/10] Single-GNN with Add fusion..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
        --model_id ADFD-SingleGNNAdd --model TAGNet --data ADFD \
        --e_layers 6 --batch_size 64 --d_model 256 --d_ff 512 --n_heads 8 \
        --resolution_list 4,8,16 --nodedim 8 --augmentations none \
        --single_gnn --single_gnn_fusion add --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [7/10]"

    echo "  [8/10] Single-GNN with Concat fusion..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
        --model_id ADFD-SingleGNNConcat --model TAGNet --data ADFD \
        --e_layers 6 --batch_size 64 --d_model 256 --d_ff 512 --n_heads 8 \
        --resolution_list 4,8,16 --nodedim 8 --augmentations none \
        --single_gnn --single_gnn_fusion concat --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [8/10]"

    echo "  [9/10] Single-GNN with Gated fusion..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
        --model_id ADFD-SingleGNNGated --model TAGNet --data ADFD \
        --e_layers 6 --batch_size 64 --d_model 256 --d_ff 512 --n_heads 8 \
        --resolution_list 4,8,16 --nodedim 8 --augmentations none \
        --single_gnn --single_gnn_fusion gated --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [9/10]"

    echo "  [10/10] Single-GNN with Hint fusion..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/ADFD/ \
        --model_id ADFD-SingleGNNHint --model TAGNet --data ADFD \
        --e_layers 6 --batch_size 64 --d_model 256 --d_ff 512 --n_heads 8 \
        --resolution_list 4,8,16 --nodedim 8 --augmentations none \
        --single_gnn --single_gnn_fusion hint --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [10/10]"

    echo "[Part 3] All experiments completed on GPU 3"
}

# ==================== Run All Parts in Parallel ====================
echo ""
echo "Starting all 3 parts in parallel..."
echo ""

# Run each part in background
run_part1 &
pid1=$!
run_part2 &
pid2=$!
run_part3 &
pid3=$!

# Wait for all parts to finish
wait $pid1
echo "Part 1 finished (Branch ablation)"

wait $pid2
echo "Part 2 finished (Channel-Resolution ablation)"

wait $pid3
echo "Part 3 finished (Dual-Graph ablation)"

echo ""
echo "=========================================="
echo "All 10 ADFD ablation experiments completed!"
echo "=========================================="