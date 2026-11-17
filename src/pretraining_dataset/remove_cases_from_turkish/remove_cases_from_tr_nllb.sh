#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 32
#SBATCH --array 0-3
#SBATCH --output logs/%A_%a_remove_cases_from_tr_nllb.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

export HF_HUB_DISABLE_XET=1
# HF_TOKEN is expected to be exported from ~/.bashrc

MODES=(case postp nominal nominal_infl)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
D=/hnvme/workspace/b279bb15-morph/Morphological/data/trilingual_100M_en_tr_de

python3 /hnvme/workspace/b279bb15-morph/Morphological/src/pretraining_dataset/remove_cases_from_turkish/remove_cases_from_tr_nllb.py \
   --input $D/trilingual_100M.tr \
   --output $D/trilingual_100M_stripped_${MODE}.tr \
   --mode $MODE \
   --num-workers $((SLURM_CPUS_PER_TASK - 2)) \
   --tag-batch-size 128