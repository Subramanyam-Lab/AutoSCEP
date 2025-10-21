#!/bin/bash
#SBATCH --job-name=worker_labeling
#SBATCH --output=logs_labeling/worker-%A_%a.out
#SBATCH --error=logs_labeling/worker-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5 
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=1-35
#SBATCH --mail-user=tzk5446@psu.edu  
#SBATCH --mail-type=FAIL             

echo "Worker ${SLURM_ARRAY_TASK_ID} proceeding with task execution."
echo "Received parameters: NUMSAM=${NUMSAM}, SEED=${SEED}"

source ~/.bashrc
conda activate myenv
module load gurobi/10.0.3
export PYTHONPATH="${PYTHONPATH}:$(dirname $0)/../src"

TASK_FILE="tasks.txt"
NUM_WORKERS=${SLURM_ARRAY_TASK_MAX}
WORKER_ID=${SLURM_ARRAY_TASK_ID}

awk -v num_workers="$NUM_WORKERS" -v worker_id="$WORKER_ID" 'NR % num_workers == (worker_id - 1)' "$TASK_FILE" | while read -r FILE_NUM PERIOD; do
    
    echo "Worker ${WORKER_ID} processing task: file_num=${FILE_NUM}, period=${PERIOD}" 
    python ../src/label_generation_adaptive.py \
        --file_num "$FILE_NUM" \
        --period "$PERIOD" \
        --num_cpus "$SLURM_CPUS_PER_TASK" \
        --numsam "$NUMSAM"\
        --seed "$SEED"
    
    echo "Worker ${WORKER_ID} finished task: file_num=${FILE_NUM}, period=${PERIOD}"
done

echo "Worker ${WORKER_ID} has no more tasks."

