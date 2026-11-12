#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:2
#SBATCH --output logs/%j_%x.out
#SBATCH --time 24:00:00

# Usage (run from the repo root, so that logs/ exists):
#   sbatch -J tr_phase2_nominal_infl src/pretraining/train_tr_stripped_phase2_regular.sh stripped_nominal_infl
#   sbatch -J tr_phase2_postp        src/pretraining/train_tr_stripped_phase2_regular.sh stripped_postp
#   sbatch -J tr_phase2_case         src/pretraining/train_tr_stripped_phase2_regular.sh stripped_case
#   sbatch -J tr_phase2_nominal      src/pretraining/train_tr_stripped_phase2_regular.sh stripped_nominal
# (-J sets the job name, which %x puts in the log filename.)
#
# Multi-seed Phase 2 (German, resume from TR Phase-1 epoch-5 of the same seed):
#   - LR = 1e-4, 5 epochs, checkpoint after every epoch, 5 seeds
#
# Checkpoint layout under --out:
#   ep{E}/seed_{S}/

MODE="${1:?Usage: sbatch $0 <mode>   (stripped_nominal_infl | stripped_postp | stripped_case | stripped_nominal)}"
case "${MODE}" in
    stripped_nominal_infl|stripped_postp|stripped_case|stripped_nominal) ;;
    *) echo "Unknown mode: ${MODE}"; exit 1 ;;
esac

SEEDS=(0 1 2 3 4)
NGPUS=2   # keep in sync with --gres gpu:N above

source ~/.bashrc
conda activate morph2

ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"
PHASE2_DATA="${ROOT}/data/NLLB_100M/NLLB_100M.de"
TR_OUT="${ROOT}/models/v5/char-tr-de"

# From REGULAR Phase 1 (for SHUFFLED Phase 1, change "regular" to "shuffled" in CKPT only):
TR_OUT_PHASE2="${TR_OUT}/phase_2/${MODE}/regular/"
PHASE_SUBDIR="phase_2/${MODE}/regular"
RUN_NAME="char-tr-de_${PHASE_SUBDIR//\//_}"

cd "${ROOT}/src"
echo "Starting multi-seed pre-training | mode=${MODE} | $(date)"

# ---------------------------------------------------------------------------
# Phase 2 — German from TR Phase-1 epoch 5 (same seed)
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    CKPT="${TR_OUT}/phase_1/${MODE}/regular/ep5/seed_${SEED}/"
    echo "=== Phase 2 DE from TR ep5 | mode=${MODE} | seed=${SEED} | ckpt=${CKPT} | $(date) ==="
    torchrun --standalone --nproc_per_node="${NGPUS}" pretraining/train.py \
        --resume_from "${CKPT}" \
        --phase2_path "${PHASE2_DATA}" \
        --tokenizer_path "${TOKENIZER}" \
        --l1 tur \
        --l2 deu \
        --out "${TR_OUT_PHASE2}" \
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
        --bf16 || { echo "TRAINING FAILED seed=${SEED}"; exit 1; }
done

echo "Finished multi-seed pre-training at $(date)"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
DATIVE_JSON="${ROOT}/src/eval_dataset/testset/dative_accusative_pairs.json"
ACCUSATIVE_JSON="${ROOT}/src/eval_dataset/testset/accusative_dative_pairs.json"
JSON_PATHS="${DATIVE_JSON},${ACCUSATIVE_JSON}"
OUT_ROOT="${ROOT}/eval_results"
EVAL_SEEDS="0,1,2,3,4"
EPOCHS="1,3,5"

mkdir -p "${OUT_ROOT}" "${ROOT}/src/eval/logs"

echo "=== Evaluating char-tr-de | ${PHASE_SUBDIR} | epochs=${EPOCHS} seeds=${EVAL_SEEDS} | $(date) ==="
python3 "${ROOT}/src/eval/evaluate_ppl.py" \
    --model_root "${TR_OUT}" \
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
