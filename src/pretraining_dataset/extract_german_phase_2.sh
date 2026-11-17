#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_extract_german_phase_2.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

echo "Starting German extraction at $(date)"

python3 pretraining_dataset/extract_german_phase_2.py \
    --original-bytes 582_042_118 \
    --original-mean 126.02 \
    --original-median 113.00 \
    --original-std 55.91 \
    --byte-premium 1.0537171

echo "Extraction completed at $(date)"