#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=25GB 
#SBATCH --time=12:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=logs/empire-%A_%a.out
#SBATCH --error=logs/empire-%A_%a.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=1-240%60

source ~/.bashrc
source activate neurmhsp
module load gurobi/10.0.3

# azs7266_sc

LENREG_LIST=(1 2 4 8 16 32 64 128)
SEED_LIST=( {1..30} )

TOTAL_LENREG=${#LENREG_LIST[@]}   # 8
TOTAL_SEEDS=${#SEED_LIST[@]}     # 30

INDEX=$(( SLURM_ARRAY_TASK_ID - 1 ))

# seed_index: 0…29, lenreg_index: 0…7
seed_index=$(( INDEX / TOTAL_LENREG ))
lenreg_index=$(( INDEX % TOTAL_LENREG ))

seed=${SEED_LIST[seed_index]}
lenreg=${LENREG_LIST[lenreg_index]}

echo "Job $SLURM_ARRAY_TASK_ID → seed=$seed, lenreg=$lenreg"


/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python diff_h_anal.py --seed $seed --lenreg $lenreg



