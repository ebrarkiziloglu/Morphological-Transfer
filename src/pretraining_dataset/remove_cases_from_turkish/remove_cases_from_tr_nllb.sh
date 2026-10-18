#!/bin/bash
#SBATCH --partition preempt
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_remove_cases_from_tr_nllb.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

# Compute nodes cannot reach Hugging Face Xet CAS; use HTTPS + the login-node cache.
export HF_HUB_DISABLE_XET=1
# HF_TOKEN is expected to be exported from ~/.bashrc

python3 pretraining_dataset/remove_cases_from_turkish/remove_cases_from_tr_nllb.py \
   --input /hnvme/workspace/b279bb15-morph/Morphological/data/NLLB_100M/NLLB_100M.tr \
   --mode case
   # --output /hnvme/workspace/b279bb15-morph/Morphological/data/NLLB_100M/NLLB_100M_case.tr

# other modes: add --mode postp, --mode nominal or --mode nominal_infl (and a matching --output)

python3 pretraining_dataset/remove_cases_from_turkish/remove_cases_from_tr_nllb.py \
   --input /hnvme/workspace/b279bb15-morph/Morphological/data/NLLB_100M/NLLB_100M.tr \
   --mode nominal

python3 pretraining_dataset/remove_cases_from_turkish/remove_cases_from_tr_nllb.py \
   --input /hnvme/workspace/b279bb15-morph/Morphological/data/NLLB_100M/NLLB_100M.tr \
   --mode postp
   
python3 pretraining_dataset/remove_cases_from_turkish/remove_cases_from_tr_nllb.py \
   --input /hnvme/workspace/b279bb15-morph/Morphological/data/NLLB_100M/NLLB_100M.tr \
   --mode nominal_infl
