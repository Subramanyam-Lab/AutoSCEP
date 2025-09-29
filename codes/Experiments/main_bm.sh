#!/bin/bash
#SBATCH --nodes=1                          
#SBATCH --ntasks=5                         
#SBATCH --cpus-per-task=8                  
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --job-name=bd_empire
#SBATCH --output=logs/ph_%a.out  
#SBATCH --error=logs/ph_%a.err   
#SBATCH --array=0-9

METHOD="PH"
SEEDS=(11 12 13 14 15 16 17 18 19 20)
TIMES=(60 300 600 1800 3600)

NUM_SEEDS=${#SEEDS[@]}

SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
TIME_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

SEED=${SEEDS[$SEED_INDEX]}
TIME=${TIMES[$TIME_INDEX]}


source ~/.bashrc
conda activate myenv
module purge
module load gurobi/10.0.3
module load gcc/13.2.0
module load openmpi/4.1.1-pmi2

export MPICH_ASYNC_PROGRESS=1

echo "Job Array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "SEED=$SEED, NUMSCE=$SLURM_NTASKS, METHOD=$METHOD, TIMELIMIT=$TIME"

srun python empire_bm.py --seed $SEED --num-sce $SLURM_NTASKS --method $METHOD --time $TIME