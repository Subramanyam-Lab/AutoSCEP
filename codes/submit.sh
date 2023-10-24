#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=neurmhsp.out
#SBATCH --error=neurmhsp.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc

source activate neurmhsp

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python CPLP.py 

