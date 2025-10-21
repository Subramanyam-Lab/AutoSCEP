#!/bin/bash
#SBATCH --job-name=adaptive_conv
#SBATCH --output=logs/adaptive_%A_%a.out
#SBATCH --error=logs/adaptive_%A_%a.err
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --account=azs7266_p_gpu
#SBATCH --partition=sla-prio

mkdir -p sampling_convergence_controlled

PERIOD_VALUES=(1 2 3 4 5 6 7 8)
MASTER_SEEDS=(42 123 456 789 1011)

PERIOD_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

PERIOD=${PERIOD_VALUES[$PERIOD_IDX]}
MASTER_SEED=${MASTER_SEEDS[$SEED_IDX]}

echo "========================================"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Period=$PERIOD, MasterSeed=$MASTER_SEED"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "========================================"

source ~/.bashrc
conda activate myenv
module load gurobi/10.0.3

python label_generation_parallel_adaptive.py \
    --period $PERIOD \
    --master_seed $MASTER_SEED \
    --num_cpus $SLURM_CPUS_PER_TASK \
    --initial_L 6

echo "End: $(date)"