#!/bin/bash
#SBATCH --partition preempt
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_shuffle-NLLB-100M-TR-Stripped.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

set -euo pipefail

IN_DIR=/hnvme/workspace/b279bb15-morph/Morphological/data/NLLB_100M
OUT_DIR=/hnvme/workspace/b279bb15-morph/Morphological/data/NLLB_100M_shuffled

DIST=geometric
RATE=0.3
GEOM_P=0.5
MAX_D=20
SEED=42

mkdir -p "${OUT_DIR}"

python3 /hnvme/workspace/b279bb15-morph/Morphological/src/pretraining_dataset/shuffle_words.py \
    --input  "${IN_DIR}/NLLB_100M_stripped_nominal_infl.tr" \
    --output_dir "${OUT_DIR}" \
    --tokenize       "whitespace" \
    --corruption_rate "${RATE}" \
    --distribution   "${DIST}" \
    --max_distance   "${MAX_D}" \
    --geom_p         "${GEOM_P}" \
    --seed           "${SEED}" \
    --save_config


python3 /hnvme/workspace/b279bb15-morph/Morphological/src/pretraining_dataset/shuffle_words.py \
    --input  "${IN_DIR}/NLLB_100M_stripped_postp.tr" \
    --output_dir "${OUT_DIR}" \
    --tokenize       "whitespace" \
    --corruption_rate "${RATE}" \
    --distribution   "${DIST}" \
    --max_distance   "${MAX_D}" \
    --geom_p         "${GEOM_P}" \
    --seed           "${SEED}" \
    --save_config

python3 /hnvme/workspace/b279bb15-morph/Morphological/src/pretraining_dataset/shuffle_words.py \
    --input  "${IN_DIR}/NLLB_100M_stripped_case.tr" \
    --output_dir "${OUT_DIR}" \
    --tokenize       "whitespace" \
    --corruption_rate "${RATE}" \
    --distribution   "${DIST}" \
    --max_distance   "${MAX_D}" \
    --geom_p         "${GEOM_P}" \
    --seed           "${SEED}" \
    --save_config

python3 /hnvme/workspace/b279bb15-morph/Morphological/src/pretraining_dataset/shuffle_words.py \
    --input  "${IN_DIR}/NLLB_100M_stripped_nominal.tr" \
    --output_dir "${OUT_DIR}" \
    --tokenize       "whitespace" \
    --corruption_rate "${RATE}" \
    --distribution   "${DIST}" \
    --max_distance   "${MAX_D}" \
    --geom_p         "${GEOM_P}" \
    --seed           "${SEED}" \
    --save_config