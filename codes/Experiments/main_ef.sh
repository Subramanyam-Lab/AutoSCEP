#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=logs/ef_%a.out
#SBATCH --error=logs/ef_%a.err
#SBATCH --account=azs7266_p_gpu
#SBATCH --partition=sla-prio
#SBATCH --array=1

source ~/.bashrc

conda activate myenv
module purge
module load gurobi/10.0.3
module load cuda/11.5.0
module load gcc/13.2.0
module load openmpi/4.1.1-pmi2

export MPICH_ASYNC_PROGRESS=1
echo "Running on nodes: $SLURM_NODELIST (Array Task: $SLURM_ARRAY_TASK_ID)"

seeds=(11) #  12 13 14 15 16 17 18 19 20
numsces=(5) #  10 20

idx=$(( SLURM_ARRAY_TASK_ID - 1 ))

SEED=${seeds[$(( idx % ${#seeds[@]} ))]}
NUMSCE=${numsces[$(( idx / ${#seeds[@]} ))]}

echo "SEED=$SEED, NUMSCE=$NUMSCE"

python empire_ef.py --seed $SEED --num-sce $NUMSCE