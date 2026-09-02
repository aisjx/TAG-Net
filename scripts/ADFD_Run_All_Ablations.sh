export CUDA_VISIBLE_DEVICES=0,1,2,3

# ==========================================
# ADFD Dataset - Run All Ablation Experiments
# Total: 10 experiments
# ==========================================

echo "=========================================="
echo "Starting ADFD Ablation Experiments"
echo "Total: 10 experiments"
echo "=========================================="

# ==================== Part 1: Branch Ablation ====================
echo ""
echo "==================== Part 1: Branch Ablation ====================="

echo "[1/10] Ablation: w/o Temporal Branch..."
bash scripts/ADFD_Ablation_Temporal.sh
echo "Done [1/10]"

echo "[2/10] Ablation: w/o Frequency Branch..."
bash scripts/ADFD_Ablation_Frequency.sh
echo "Done [2/10]"

# ==================== Part 2: Channel-Aware Resolution Encoder Ablation ====================
echo ""
echo "========== Part 2: Channel-Aware Resolution Encoder Ablation =========="

echo "[3/10] Ablation: w/o Channel Branch..."
bash scripts/ADFD_Ablation_ChannelBranch.sh
echo "Done [3/10]"

echo "[4/10] Ablation: w/o Resolution Router..."
bash scripts/ADFD_Ablation_ResolutionRouter.sh
echo "Done [4/10]"

echo "[5/10] Ablation: w/o Channel-Aware Multi-Resolution Encoder..."
bash scripts/ADFD_Ablation_ChannelResolution.sh
echo "Done [5/10]"

# ==================== Part 3: Dual-Graph Interaction Ablation ====================
echo ""
echo "============ Part 3: Dual-Graph Interaction Ablation ============"

echo "[6/10] Ablation: w/o Dual-Graph Interaction..."
bash scripts/ADFD_Ablation_DualGraph.sh
echo "Done [6/10]"

echo "[7/10] Single-GNN with Add fusion..."
bash scripts/ADFD_Ablation_Add.sh
echo "Done [7/10]"

echo "[8/10] Single-GNN with Concat fusion..."
bash scripts/ADFD_Ablation_Concat.sh
echo "Done [8/10]"

echo "[9/10] Single-GNN with Gated fusion..."
bash scripts/ADFD_Ablation_Gated.sh
echo "Done [9/10]"

echo "[10/10] Single-GNN with Hint fusion..."
bash scripts/ADFD_Ablation_Hint.sh
echo "Done [10/10]"

echo ""
echo "=========================================="
echo "All 10 ADFD ablation experiments completed!"
echo "=========================================="