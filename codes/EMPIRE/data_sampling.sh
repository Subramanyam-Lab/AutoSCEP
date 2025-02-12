#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB 
#SBATCH --time=12:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=empire-%A_%a.out
#SBATCH --error=empire-%A_%a.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=0

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3

# SEED=$SLURM_ARRAY_TASK_ID
PROB=$SLURM_ARRAY_TASK_ID

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python FSD_sampling3.py --prob $PROB