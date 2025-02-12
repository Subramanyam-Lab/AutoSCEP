#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB 
#SBATCH --time=12:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=empire-%j.out
#SBATCH --error=empire-%j.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=40

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3

SEED=$SLURM_ARRAY_TASK_ID

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python empire_embedding_main2.py --seed $SEED
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python ml_model_training.py
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python FSD_progressive_hedging.py