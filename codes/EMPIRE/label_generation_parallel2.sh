#!/bin/bash
#SBATCH --job-name=worker_labeling
#SBATCH --output=logs3/worker-%A_%a.out
#SBATCH --error=logs3/worker-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5 
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=1-35%35  # ★★★ 동시에 실행할 워커 수 (예: 200개)
#SBATCH --mail-user=tzk5446@psu.edu  
#SBATCH --mail-type=FAIL             

source ~/.bashrc
conda activate neurmhsp
module load gurobi/10.0.3

# TASK_FILE="tasks.txt"
# NUM_WORKERS=${SLURM_ARRAY_TASK_MAX}
# WORKER_ID=${SLURM_ARRAY_TASK_ID}

# awk -v num_workers="$NUM_WORKERS" -v worker_id="$WORKER_ID" 'NR % num_workers == (worker_id - 1)' "$TASK_FILE" | while read -r FILE_NUM PERIOD; do
    
#     echo "Worker ${WORKER_ID} processing task: file_num=${FILE_NUM}, period=${PERIOD}"

#     /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python label_generation_parallel.py \
#         --file_num "$FILE_NUM" \
#         --period "$PERIOD" \
#         --num_cpus "$SLURM_CPUS_PER_TASK"
    
#     echo "Worker ${WORKER_ID} finished task: file_num=${FILE_NUM}, period=${PERIOD}"
# done

# echo "Worker ${WORKER_ID} has no more tasks."



TASK_FILE="tasks.txt"
START_FILE_NUM=2949 
NUM_WORKERS=${SLURM_ARRAY_TASK_MAX}
WORKER_ID=${SLURM_ARRAY_TASK_ID}

awk -v num_workers="$NUM_WORKERS" -v worker_id="$WORKER_ID" -v start_num="$START_FILE_NUM" \
    '$1 > start_num && $1 <= 5000 && NR % num_workers == (worker_id - 1)' "$TASK_FILE" | while read -r FILE_NUM PERIOD; do
    
    echo "Worker ${WORKER_ID} processing task: file_num=${FILE_NUM}, period=${PERIOD}"

    /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python label_generation_parallel_fixed.py \
        --file_num "$FILE_NUM" \
        --period "$PERIOD" \
        --num_cpus "$SLURM_CPUS_PER_TASK"
    
    echo "Worker ${WORKER_ID} finished task: file_num=${FILE_NUM}, period=${PERIOD}"
done

echo "Worker ${WORKER_ID} has no more tasks."