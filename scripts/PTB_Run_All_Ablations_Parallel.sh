#!/bin/bash

# ==========================================
# PTB Dataset - Run All Ablations (Parallel)
# Part 1 (Branch)      → GPU 0
# Part 2 (ChannelRes)  → GPU 1
# Part 3 (DualGraph)   → GPU 2
# ==========================================

echo "=========================================="
echo "Starting PTB Ablation Experiments (Parallel)"
echo "Part 1 (Branch)     → GPU 0"
echo "Part 2 (ChannelRes) → GPU 1"
echo "Part 3 (DualGraph)  → GPU 2"
echo "=========================================="

# ==================== Part 1: Branch Ablation (GPU 0) ====================
run_part1() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[Part 1] Started on GPU 0"

    echo "  [1/10] w/o Temporal Branch..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
        --model_id PTB-AblTemp --model TAGNet --data PTB \
        --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 --n_heads 8 \
        --resolution_list 2,4,8,16 --nodedim 8 --augmentations drop0.2 \
        --disable_temporal_branch --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [1/10]"

    echo "  [2/10] w/o Frequency Branch..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
        --model_id PTB-AblFreq --model TAGNet --data PTB \
        --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 --n_heads 8 \
        --resolution_list 2,4,8,16 --nodedim 8 --augmentations drop0.2 \
        --disable_frequency_branch --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [2/10]"

    echo "[Part 1] All experiments completed on GPU 0"
}

# ==================== Part 2: Channel-Aware Resolution Encoder Ablation (GPU 1) ====================
run_part2() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[Part 2] Started on GPU 1"

    echo "  [3/10] w/o Channel Branch..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
        --model_id PTB-AblChanBranch --model TAGNet --data PTB \
        --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 --n_heads 8 \
        --resolution_list 2,4,8,16 --nodedim 8 --augmentations drop0.2 \
        --disable_channel_branch --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [3/10]"

    echo "  [4/10] w/o Resolution Router..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
        --model_id PTB-AblResRouter --model TAGNet --data PTB \
        --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 --n_heads 8 \
        --resolution_list 2,4,8,16 --nodedim 8 --augmentations drop0.2 \
        --disable_resolution_router --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [4/10]"

    echo "  [5/10] w/o Channel-Aware Multi-Resolution Encoder..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
        --model_id PTB-AblChanRes --model TAGNet --data PTB \
        --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 --n_heads 8 \
        --resolution_list 2,4,8,16 --nodedim 8 --augmentations drop0.2 \
        --disable_channel_resolution_module --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [5/10]"

    echo "[Part 2] All experiments completed on GPU 1"
}

# ==================== Part 3: Dual-Graph Interaction Ablation (GPU 2) ====================
run_part3() {
    export CUDA_VISIBLE_DEVICES=2
    echo "[Part 3] Started on GPU 2"

    echo "  [6/10] w/o Dual-Graph Interaction..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
        --model_id PTB-AblDualGraph --model TAGNet --data PTB \
        --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 --n_heads 8 \
        --resolution_list 2,4,8,16 --nodedim 8 --augmentations drop0.2 \
        --disable_cross_graph_interaction --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [6/10]"

    echo "  [7/10] Single-GNN with Add fusion..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
        --model_id PTB-SingleGNNAdd --model TAGNet --data PTB \
        --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 --n_heads 8 \
        --resolution_list 2,4,8,16 --nodedim 8 --augmentations drop0.2 \
        --single_gnn --single_gnn_fusion add --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [7/10]"

    echo "  [8/10] Single-GNN with Concat fusion..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
        --model_id PTB-SingleGNNConcat --model TAGNet --data PTB \
        --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 --n_heads 8 \
        --resolution_list 2,4,8,16 --nodedim 8 --augmentations drop0.2 \
        --single_gnn --single_gnn_fusion concat --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [8/10]"

    echo "  [9/10] Single-GNN with Gated fusion..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
        --model_id PTB-SingleGNNGated --model TAGNet --data PTB \
        --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 --n_heads 8 \
        --resolution_list 2,4,8,16 --nodedim 8 --augmentations drop0.2 \
        --single_gnn --single_gnn_fusion gated --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [9/10]"

    echo "  [10/10] Single-GNN with Hint fusion..."
    python -u run.py \
        --task_name classification --is_training 1 \
        --root_path /home/ljh2025/ljh/DMKformer0/dataset/PTB/ \
        --model_id PTB-SingleGNNHint --model TAGNet --data PTB \
        --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 --n_heads 8 \
        --resolution_list 2,4,8,16 --nodedim 8 --augmentations drop0.2 \
        --single_gnn --single_gnn_fusion hint --des 'Exp' --itr 3 \
        --learning_rate 0.0003 --train_epochs 15 --patience 8 --gpu 0
    echo "  Done [10/10]"

    echo "[Part 3] All experiments completed on GPU 2"
}

# ==================== Run All Parts in Parallel ====================
echo ""
echo "Starting all 3 parts in parallel..."
echo ""

run_part1 &
pid1=$!
run_part2 &
pid2=$!
run_part3 &
pid3=$!

wait $pid1
echo "Part 1 finished (Branch ablation)"

wait $pid2
echo "Part 2 finished (Channel-Resolution ablation)"

wait $pid3
echo "Part 3 finished (Dual-Graph ablation)"

echo ""
echo "=========================================="
echo "All 10 PTB ablation experiments completed!"
echo "=========================================="