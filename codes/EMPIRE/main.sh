#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100GB 
#SBATCH --time=12:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=empire_log/empire-%A_%a.out
#SBATCH --error=empire_log/empire-%A_%a.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=100

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3

# SEED=$SLURM_ARRAY_TASK_ID
PROB=$SLURM_ARRAY_TASK_ID
SEED=$SLURM_ARRAY_TASK_ID

# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python empire_embedding_main3.py --seed $SEED
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python FSD_sampling3.py --prob $PROB
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python Data_build_i_3.py
# /storage/home/tzk5446/.conda/envs/project1/bin/python model_training3.py
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python reduced_run.py
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python Random_sce_generator.py
/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python REDUCED_NEUREMPIRE.py --seed $SEED