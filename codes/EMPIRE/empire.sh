#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=180GB 
#SBATCH --time=12:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=empire-%j.out
#SBATCH --error=empire-%j.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python empire_embedding_main.py 


