#!/bin/bash
#SBATCH --partition preempt
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_extract_nllb_german_%d.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

echo "Starting German extraction at $(date)"
# HF_TOKEN is expected to be exported from ~/.bashrc

python3 pretraining_dataset/nllb_subset/extract_nllb_german.py --token $HF_TOKEN \
    --original-bytes 596_173_714 \
    --original-mean 126.02 \
    --original-median 113.00 \
    --original-std 55.91 \
    --byte-premium 1.0537171

echo "Extraction completed at $(date)"