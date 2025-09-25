#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --mem=400GB 
#SBATCH --time=24:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=log/empire-%A.out
#SBATCH --error=log/empire-%A.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3


/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python experiments_for_label.py