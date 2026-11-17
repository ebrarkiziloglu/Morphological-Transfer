#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:2
#SBATCH --output logs/%A_%a_de_phase1_shuffled.out
#SBATCH --time 24:00:00
#SBATCH --array=0-4   # one job per seed (seed = SLURM_ARRAY_TASK_ID)
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
#   - LR = 1e-4, 5 epochs
#   - checkpoint after every epoch
#   - 5 seeds

source ~/.bashrc
conda activate morph2

echo "Starting pre-training | seed=${SEED} at $(date)"
 
ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"

# ---------------------------------------------------------------------------
# Phase 1 — German (NLLB 100M)
# ---------------------------------------------------------------------------

# REGULAR:
# DE_P1="${ROOT}/data/trilingual_100M_en_tr_de/trilingual_100M.de"
# DE_OUT="${ROOT}/models/v5/char-de/phase_1/regular/"

# SHUFFLED:
DE_P1="${ROOT}/data/trilingual_100M_en_tr_de/trilingual_100M_shuffled_geometric_r0.3.de"
DE_OUT="${ROOT}/models/v5/char-de/phase_1/shuffled/"

SEED="${SLURM_ARRAY_TASK_ID:?submit with sbatch (array job); seed = array task id}"
if [ -z "${EVAL_ONLY}" ]; then
    echo "=== Phase 1 German | seed=${SEED} | $(date) ==="
    torchrun --standalone --nproc_per_node="${NGPUS}" pretraining/train.py \
        --tokenizer_path "${TOKENIZER}" \
        --phase1_path "${DE_P1}" \
        --l1 deu \
        --l2 deu \
        --out "${DE_OUT}" \
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
fi

echo "Finished seed ${SEED} at $(date)"
