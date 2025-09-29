#!/bin/bash
#SBATCH --job-name=neurumhsp_parallel
#SBATCH --output=logs/parallel_labeling-%A_%a.out
#SBATCH --error=logs/parallel_labeling-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5 
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=1-800%200
#SBATCH --mail-user=tzk5446@psu.edu  
#SBATCH --mail-type=FAIL             


source ~/.bashrc
conda activate neurmhsp
module load gurobi/10.0.3

TOTAL_PERIODS=8
FILE_NUM=$(( (SLURM_ARRAY_TASK_ID - 1) / TOTAL_PERIODS + 1 ))
PERIOD=$(( (SLURM_ARRAY_TASK_ID - 1) % TOTAL_PERIODS + 1 ))

echo "Running SLURM array task ${SLURM_ARRAY_TASK_ID} with ${SLURM_CPUS_PER_TASK} CPUs."
echo "Corresponds to file_num=${FILE_NUM}, period=${PERIOD}"

/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python  label_generation_parallel.py \
    --file_num $FILE_NUM \
    --period $PERIOD \
    --num_cpus $SLURM_CPUS_PER_TASK