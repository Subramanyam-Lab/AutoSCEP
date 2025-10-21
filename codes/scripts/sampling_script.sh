#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=8  
#SBATCH --time=12:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=log/full_sampling-%j.out
#SBATCH --error=log/full_sampling-%j.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=1

source ~/.bashrc

source activate myenv
module load gurobi/10.0.3

echo "Starting sampling with NUMSAM=${NUMSAM}, SEED=${SEED}"
export PYTHONPATH="${PYTHONPATH}:$(dirname $0)/../src"

python ../src/sampling.py --num_samples $NUMSAM --seed $SEED

echo "Sampling finished."