#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128GB 
#SBATCH --time=24:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=log/empire-%A_%a.out
#SBATCH --error=log/empire-%A_%a.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=1

source ~/.bashrc

source activate neurmhsp
module load gurobi/10.0.3

# azs7266_sc
# SEED=$SLURM_ARRAY_TASK_ID

PROB=$SLURM_ARRAY_TASK_ID

SEED=$SLURM_ARRAY_TASK_ID
SOLNUM=1
LENREG=1


# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python empire_embedding_main4.py --seed $SEED
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python model_check.py --seed $SEED
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python Random_sce_generator.py
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python FSD_sampling3.py --prob $PROB 
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python Data_build_i_3.py
# /storage/home/tzk5446/.conda/envs/project1/bin/python model_training3.py
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python reduced_run.py
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python REDUCED_NEUREMPIRE.py --seed $SEED
# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python Data_preprocessing.py
/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python post_process_data.py

