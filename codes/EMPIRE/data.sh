#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=empire-%A_%a.out
#SBATCH --error=empire-%A_%a.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=1-8 

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3

PERIOD=$SLURM_ARRAY_TASK_ID

echo "Running for PERIOD=$PERIOD"

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python Data_generation_run.py --period $PERIOD