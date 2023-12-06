#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --job-name=NNtrain
#SBATCH --output=NNtrain1.out
#SBATCH --error=NNtrain1.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc

source activate neurmhsp

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python Neural2SP_V2.py 

