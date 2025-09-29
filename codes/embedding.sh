#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB 
#SBATCH --time=24:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=log/embedding_run-%j.out
#SBATCH --error=log/embedding_run-%j.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc

source activate myenv
module load gurobi/10.0.3

echo "Starting ML Embedding and Solving with NUMSAM=${NUMSAM}, SEED=${SEED}"

python embedding_main.py --num_samples $NUMSAM --seed $SEED

