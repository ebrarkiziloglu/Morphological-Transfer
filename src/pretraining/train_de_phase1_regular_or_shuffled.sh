#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_de_phase1_regular.out
#SBATCH --time 24:00:00
SEEDS=(0 1 2 3 4)

# Multi-seed two-phase pretraining
# --------------------------------
# Phase 1 (EN, TR and ZH separately):
#   - LR = 1e-4, 5 epochs
#   - checkpoints after epochs 5
#   - 5 seeds
# Phase 2 (German, resume from Phase-1 epoch-5 of the same seed):
#   - LR = 5e-5, 5 epochs
#   - checkpoint after every epoch
#   - 5 seeds

source ~/.bashrc
conda activate morph2

echo "Starting multi-seed pre-training at $(date)"
 
ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"

# ---------------------------------------------------------------------------
# Phase 1 — German (NLLB 100M)
# ---------------------------------------------------------------------------

# REGULAR:
DE_P1="${ROOT}/data/NLLB_100M/NLLB_100M.de"
DE_OUT="${ROOT}/models/v4/char-de/phase_1/regular/"

# SHUFFLED:
# DE_P1="${ROOT}/data/NLLB_100M_shuffled/NLLB_100M_shuffled_geometric_r0.3.de"
# DE_OUT="${ROOT}/models/v4/char-de/phase_1/shuffled/"

for SEED in "${SEEDS[@]}"; do
    echo "=== Phase 1 German | seed=${SEED} | $(date) ==="
    python3 pretraining/train.py \
        --tokenizer_path "${TOKENIZER}" \
        --phase1_path "${DE_P1}" \
        --l1 eng \
        --l2 deu \
        --out "${DE_OUT}" \
        --max_seq_len 1024 \
        --max_position_embeddings 1024 \
        --batch_size 16 \
        --epochs_phase1 5 \
        --epochs_phase2 0 \
        --save_epochs_phase1 5 \
        --lr 1e-4 \
        --grad_acc 2 \
        --logging_steps 100 \
        --seed "${SEED}" \
        --attn_implementation flash_attention_3 \
        --bf16
done

echo "Finished multi-seed pre-training at $(date)"
