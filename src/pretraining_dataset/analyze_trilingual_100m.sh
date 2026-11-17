#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_analyze_trilingual_100m.out
#SBATCH --time 04:00:00

source ~/.bashrc
conda activate morph2

# dataset_analysis_helper.py lives one level up in pretraining_dataset/
export PYTHONPATH="$PWD/pretraining_dataset:$PYTHONPATH"
python pretraining_dataset/analyze_trilingual_100m.py
