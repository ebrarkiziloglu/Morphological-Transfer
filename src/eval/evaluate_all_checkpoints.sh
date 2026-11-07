#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output src/eval/logs/%j_eval_all_checkpoints.out
#SBATCH --time 24:00:00
#
# Evaluate EVERY checkpoint found (recursively) under a models directory.
#
# Usage:
#   sbatch src/eval/evaluate_all_checkpoints.sh <models_dir> <out_dir>
#   # split the work over several GPUs (one array task = one shard):
#   sbatch --array=0-7 src/eval/evaluate_all_checkpoints.sh <models_dir> <out_dir>
#   # afterwards (no GPU needed), build the combined CSVs:
#   bash src/eval/evaluate_all_checkpoints.sh <models_dir> <out_dir> merge
#
# Results land in <out_dir>/<path of checkpoint relative to models_dir>/
# (overall.csv, by_category.csv, per_item.csv). Finished checkpoints are skipped,
# so a preempted job can just be resubmitted.

MODELS_DIR="${1:?usage: $0 <models_dir> <out_dir> [merge]}"
OUT_DIR="${2:?usage: $0 <models_dir> <out_dir> [merge]}"
MODE="${3:-eval}"

source ~/.bashrc
conda activate morph2

# Strict mode only after conda activation (its activate.d scripts use unset vars).
set -euo pipefail

ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
cd "${ROOT}"
mkdir -p "${ROOT}/src/eval/logs" "${OUT_DIR}"

DATIVE_JSON="${ROOT}/src/eval_dataset/testset/dative_accusative_pairs.json"
ACCUSATIVE_JSON="${ROOT}/src/eval_dataset/testset/accusative_dative_pairs.json"
JSON_PATHS="${DATIVE_JSON},${ACCUSATIVE_JSON}"
TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"
# TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char_old/"

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

EXTRA=()
if [[ "${MODE}" == "merge" ]]; then
    EXTRA+=(--merge_only)
fi

echo "=== Evaluating all checkpoints under ${MODELS_DIR} | shard ${SHARD_INDEX}/${NUM_SHARDS} | mode=${MODE} | $(date) ==="
python3 "${ROOT}/src/eval/evaluate_ppl.py" \
    --scan_root "${MODELS_DIR}" \
    --tokenizer_path "${TOKENIZER}" \
    --json_path "${JSON_PATHS}" \
    --gold_key gold_text \
    --corrupt_key corrupted \
    --category_key category \
    --max_length 1024 \
    --out_dir "${OUT_DIR}" \
    --shard_index "${SHARD_INDEX}" \
    --num_shards "${NUM_SHARDS}" \
    --bootstrap_n 0 \
    "${EXTRA[@]}"

echo "Finished at $(date). Results in ${OUT_DIR}"
