#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:2
#SBATCH --output logs/%A_%a_tr_phase2_shuffled_all_seeds.out
#SBATCH --time 24:00:00
#SBATCH --array=0-4   # one job per seed (seed = SLURM_ARRAY_TASK_ID)
NGPUS=2   # keep in sync with --gres gpu:N above

# Multi-seed two-phase pretraining
# --------------------------------
# Phase 1 (EN, TR and ZH separately):
#   - LR = 1e-4, 5 epochs
#   - checkpoints after epochs 5
#   - 5 seeds
# Phase 2 (German, resume from Phase-1 epoch-5 of the same seed):
#   - LR = 1e-4, 5 epochs
#   - checkpoint after every epoch
#   - 5 seeds
#
# Checkpoint layout under --out:
#   ep{E}/seed_{S}/
# e.g. phase2/regular/regular/ep1/seed_0 ... phase2/regular/regular/ep5/seed_0

source ~/.bashrc
conda activate morph2

echo "Starting pre-training | seed=${SLURM_ARRAY_TASK_ID} at $(date)"
 
ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"
PHASE2_DATA="${ROOT}/data/NLLB_100M_phase_2/NLLB_100M.de"
cd "${ROOT}/src"   # train.py is addressed relative to src/

TR_OUT="${ROOT}/models/v5/char-tr-de"

# From REGULAR Phase 1:
# TR_OUT_PHASE2="${TR_OUT}/phase_2/regular/regular"

# From SHUFFLED Phase 1:
TR_OUT_PHASE2="${TR_OUT}/phase_2/shuffled/regular"

# ---------------------------------------------------------------------------
# Phase 2 — German from TR Phase-1 epoch 5 (same seed)
# ---------------------------------------------------------------------------
SEED="${SLURM_ARRAY_TASK_ID:?submit with sbatch (array job); seed = array task id}"
if [ -z "${EVAL_ONLY}" ]; then
    # CKPT="${TR_OUT}/phase_1/regular/ep5/seed_${SEED}/"
    CKPT="${TR_OUT}/phase_1/shuffled/ep5/seed_${SEED}/"
    echo "=== Phase 2 DE from TR ep5 | seed=${SEED} | ckpt=${CKPT} | $(date) ==="
    torchrun --standalone --nproc_per_node="${NGPUS}" pretraining/train.py \
        --resume_from "${CKPT}" \
        --phase2_path "${PHASE2_DATA}" \
        --tokenizer_path "${TOKENIZER}" \
        --l1 tur \
        --l2 deu \
        --out "${TR_OUT_PHASE2}/" \
        --max_seq_len 1024 \
        --max_position_embeddings 1024 \
        --batch_size 64 \
        --epochs_phase2 5 \
        --lr 1e-4 \
        --grad_acc 1 \
        --length_bucketing \
        --cpus 8 \
        --logging_steps 100 \
        --seed "${SEED}" \
        --attn_implementation flash_attention_3 \
        --bf16
fi

echo "Finished seed ${SEED} at $(date)"
# Evaluation covers ALL seeds, so it must run once after the whole array is
# done, not inside each task:
#   sbatch --dependency=afterok:<ARRAY_JOBID> --array=0 --gres=gpu:1 \
#          --export=ALL,EVAL_ONLY=1 <this script> [args]
[ -z "${EVAL_ONLY}" ] && exit 0
cd "${ROOT}"
DATIVE_JSON="${ROOT}/src/eval_dataset/testset/dative_accusative_pairs.json"
ACCUSATIVE_JSON="${ROOT}/src/eval_dataset/testset/accusative_dative_pairs.json"
JSON_PATHS="${DATIVE_JSON},${ACCUSATIVE_JSON}"
OUT_ROOT="${ROOT}/eval_results"
EVAL_SEEDS="0,1,2,3,4"
EPOCHS="1,3,5"
PHASE_SUBDIR="phase_2/shuffled/regular"
RUN_NAME="char-tr-de_${PHASE_SUBDIR//\//_}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${ROOT}/src/eval/logs"

TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"

echo "=== Evaluating char-tr-de | ${PHASE_SUBDIR} | epochs=${EPOCHS} seeds=${EVAL_SEEDS} | $(date) ==="
python3 ${ROOT}/src/eval/evaluate_ppl.py \
    --model_root "${ROOT}/models/v5/char-tr-de" \
    --tokenizer_path "${TOKENIZER}" \
    --run_name "${RUN_NAME}" \
    --phase_subdir "${PHASE_SUBDIR}" \
    --seeds "${EVAL_SEEDS}" \
    --epochs "${EPOCHS}" \
    --json_path "${JSON_PATHS}" \
    --gold_key gold_text \
    --corrupt_key corrupted \
    --category_key category \
    --max_length 1024 \
    --out_dir "${OUT_ROOT}/${RUN_NAME}" \
    --bootstrap_n 0

echo "Aggregate CSVs (seed averages):"
echo "  ${OUT_ROOT}/${RUN_NAME}/"
