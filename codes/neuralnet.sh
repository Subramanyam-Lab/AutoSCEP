#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --job-name=NNtrain
#SBATCH --output=NNtrain.out
#SBATCH --error=NNtrain.err
#SBATCH --account=azs7266_p_gpu
#SBATCH --partition=sla-prio

source ~/.bashrc

source activate neurmhsp

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python Neural2SP.py 

