#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:2
#SBATCH --output logs/%j_tr_phase1_shuffled.out
#SBATCH --time 24:00:00
SEEDS=(0 1 2 3 4)
NGPUS=2   # keep in sync with --gres gpu:N above

# Multi-seed two-phase pretraining
# --------------------------------
# Parallelism: DDP over NGPUS GPUs (torchrun). --batch_size is PER GPU, so the
# effective batch = batch_size * grad_acc * NGPUS = 64 * 1 * 2 = 128 sequences
# (the old single-GPU run used 16 * 2 = 32).
# The data is tokenized once and cached in <data dir>/token_cache; batches are
# padded to their longest line (not 1024) and length-bucketed.
#
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
# Phase 1 — Turkish (NLLB 10M)
# ---------------------------------------------------------------------------

# REGULAR:
# TR_P1="${ROOT}/data/NLLB_100M/NLLB_100M.tr"
# TR_OUT="${ROOT}/models/v4/char-tr-de/phase_1/regular/"

# SHUFFLED:
TR_P1="${ROOT}/data/NLLB_100M_shuffled/NLLB_100M_shuffled_geometric_r0.3.tr"
TR_OUT="${ROOT}/models/v4/char-tr-de/phase_1/shuffled/"

for SEED in "${SEEDS[@]}"; do
    echo "=== Phase 1 Turkish | seed=${SEED} | $(date) ==="
    torchrun --standalone --nproc_per_node="${NGPUS}" pretraining/train.py \
        --tokenizer_path "${TOKENIZER}" \
        --phase1_path "${TR_P1}" \
        --l1 tur \
        --l2 deu \
        --out "${TR_OUT}" \
        --max_seq_len 1024 \
        --max_position_embeddings 1024 \
        --batch_size 64 \
        --epochs_phase1 5 \
        --epochs_phase2 0 \
        --save_epochs_phase1 5 \
        --lr 1e-4 \
        --grad_acc 1 \
        --logging_steps 100 \
        --length_bucketing \
        --cpus 8 \
        --seed "${SEED}" \
        --attn_implementation flash_attention_3 \
        --bf16
done

echo "Finished multi-seed pre-training at $(date)"
