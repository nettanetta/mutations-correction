#!/usr/bin/env bash
#SBATCH --gres=gpu,vmem:22g
#SBATCH --mem=16g
#SBATCH -c 4
#SBATCH -t 1-0
#SBATCH --mail-user=netta.barak@mail.huji.ac.il
#SBATCH --mail-type=ALL

source /sci/labs/morani/morani/icore-data/lab/Tools/personal_condas/Netta/miniforge3/etc/profile.d/conda.sh
conda activate mutation_correction_env_1
module load cuda/11.8
module load nvidia

python3 -u run_detection.py
