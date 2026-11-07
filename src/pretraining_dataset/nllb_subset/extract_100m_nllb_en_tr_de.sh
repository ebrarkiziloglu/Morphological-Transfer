#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_extract_100m_nllb_en_tr_de.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

# dataset_analysis_helper.py lives one level up in pretraining_dataset/
export PYTHONPATH="$PWD/pretraining_dataset:$PYTHONPATH"
python pretraining_dataset/nllb_subset/extract_100m_nllb_en_tr_de.py
