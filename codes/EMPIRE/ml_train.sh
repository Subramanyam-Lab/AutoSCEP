#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=16GB 
#SBATCH --time=24:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=log/mltrain-%j.out
#SBATCH --error=log/mltrain-%j.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc
source activate myenv

echo "Starting ML training with NUMSAM=${NUMSAM}, SEED=${SEED}"

python ml_train.py --num_samples $NUMSAM --seed $SEED
