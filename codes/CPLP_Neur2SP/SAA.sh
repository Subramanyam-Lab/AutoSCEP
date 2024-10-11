#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --time=96:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=SAA1.out
#SBATCH --error=SAA1.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python CPLP_SAA.py 

