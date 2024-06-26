#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --time=96:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=empire.out
#SBATCH --error=empire.err
#SBATCH --account=azs7266_p_gpu
#SBATCH --partition=sla-prio
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3
module load cuda/11.5

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python run.py 

