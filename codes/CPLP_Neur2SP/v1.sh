#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=300GB
#SBATCH --time=96:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=v1.out
#SBATCH --error=v1.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python CPLP_V1.py 

