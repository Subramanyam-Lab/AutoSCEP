#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB 
#SBATCH --time=24:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=log/alternating-%A_%a.out
#SBATCH --error=log/alternating-%A_%a.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=1

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3


/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python main_alternating.py

