#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB 
#SBATCH --time=24:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=toy-%j.out
#SBATCH --error=toy-%j.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python toy3_v1.py