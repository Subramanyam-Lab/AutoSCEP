#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=complementary_relu
#SBATCH --output=logs2/train_model_%A_%a.out
#SBATCH --error=logs2/train_model_%A_%a.err
#SBATCH --account=azs7266_p_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-12

# Load environment and modules
source ~/.bashrc
conda activate neurmhsp
module load gurobi/10.0.3


export N_PROCESSES=4
export GUROBI_THREADS=7


# Set parameters based on SLURM_ARRAY_TASK_ID
declare -a N_VALUES=(50 100 200 300)
declare -a K_RATIOS=(0.25 0.5 0.75)
INSTANCES_PER_COMBINATION=1000
NUM_SAMPLES_PER_INSTANCE=100

N_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / ${#K_RATIOS[@]} ))
K_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % ${#K_RATIOS[@]} ))

N=${N_VALUES[$N_IDX]}
K_RATIO=${K_RATIOS[$K_IDX]}

echo "Starting job with n=$N, k_ratio=$K_RATIO..."

# Run the Python script with the selected parameters
/storage/home/tzk5446/.conda/envs/project1/bin/python main3.py \
  --n_values $N \
  --k_ratios $K_RATIO \
  --instances_per_combination $INSTANCES_PER_COMBINATION \
  --num_samples_per_instance $NUM_SAMPLES_PER_INSTANCE

echo "Job completed for n=$N, k_ratio=$K_RATIO."
