#!/bin/bash
#SBATCH --job-name=neurumhsp
#SBATCH --output=logs/empire-%A_%a.out
#SBATCH --error=logs/empire-%A_%a.err
#SBATCH --array=1-4000%40        # 4,000 array tasks with 400 concurrent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   
#SBATCH --mem=32GB          
#SBATCH --time=05:00:00            # Adjusted time based on expected workload
#SBATCH --account=azs7266_sc

# Load necessary modules and activate environment
source ~/.bashrc
conda activate neurmhsp
module load gurobi/10.0.3

# Calculate the range of operations for this array task
TASK_ID=${SLURM_ARRAY_TASK_ID}
OPERATIONS_PER_TASK=20
TOTAL_OPERATIONS=80000

START_INDEX=$(( (TASK_ID - 1) * OPERATIONS_PER_TASK + 1 ))
END_INDEX=$(( TASK_ID * OPERATIONS_PER_TASK ))

# Ensure END_INDEX does not exceed TOTAL_OPERATIONS
if [ "$END_INDEX" -gt "$TOTAL_OPERATIONS" ]; then
    END_INDEX=$TOTAL_OPERATIONS
fi

echo "Running task $TASK_ID: processing operations $START_INDEX to $END_INDEX"

# Execute the Python script with the calculated range and TASK_ID
/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python Data_generation_run3.py \
    --task_id $TASK_ID \
    --start_index $START_INDEX \
    --end_index $END_INDEX
