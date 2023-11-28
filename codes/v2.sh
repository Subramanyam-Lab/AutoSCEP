#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=300GB
#SBATCH --time=96:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=v2.out
#SBATCH --error=v2.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc

source activate neurmhsp

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python CPLP_V2.py 

