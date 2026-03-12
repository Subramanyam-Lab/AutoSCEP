#!/bin/bash
#SBATCH --job-name=fixed_conv
#SBATCH --output=logs/fixed_%A_%a.out
#SBATCH --error=logs/fixed_%A_%a.err
#SBATCH --array=0-639%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

mkdir -p sampling_convergence_controlled

L_VALUES=(12 24 36 48)
N_VALUES=(5 10 20 30)
PERIOD_VALUES=(1 2 3 4 5 6 7 8)
MASTER_SEEDS=(42 123 456 789 1011)

L_IDX=$((SLURM_ARRAY_TASK_ID / 160))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 160))
N_IDX=$((REMAINDER / 40))
REMAINDER=$((REMAINDER % 40))
PERIOD_IDX=$((REMAINDER / 5))
SEED_IDX=$((REMAINDER % 5))

L=${L_VALUES[$L_IDX]}
N=${N_VALUES[$N_IDX]}
PERIOD=${PERIOD_VALUES[$PERIOD_IDX]}
MASTER_SEED=${MASTER_SEEDS[$SEED_IDX]}

echo "========================================"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "L=$L, N=$N, Period=$PERIOD, MasterSeed=$MASTER_SEED"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "========================================"

source ~/.bashrc
conda activate myenv
module load gurobi/10.0.3


python label_generation_parallel_fixed.py \
    --L $L \
    --N $N \
    --period $PERIOD \
    --master_seed $MASTER_SEED \
    --num_cpus $SLURM_CPUS_PER_TASK

echo "End: $(date)"