#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=log_valid/mlvalidation/mlembed_validation_fixed-%A_%a.out
#SBATCH --error=log_valid/mlvalidation/mlembed_validation_fixed-%A_%a.err
#SBATCH --account=azs7266_p_gpu
#SBATCH --partition=sla-prio
#SBATCH --array=1-4000%40

source ~/.bashrc
conda activate neurmhsp
module load gurobi/10.0.3

methods=("MLP" "LR")
solnums=(1 2 3 4 5 6 7 8 9 10)
numsces=(1000 5000)
seeds=(11 12 13 14 15 16 17 18 19 20)
setnums=(1 2 3 4 5 6 7 8 9 10)

num_methods=${#methods[@]}
num_solnums=${#solnums[@]}
num_numsces=${#numsces[@]}
num_seeds=${#seeds[@]}
num_setnums=${#setnums[@]} 

idx=$(( SLURM_ARRAY_TASK_ID - 1 ))

seedidx=$(( idx % num_seeds ))
sceidx=$(( (idx / num_seeds) % num_numsces ))
solidx=$(( (idx / (num_seeds * num_numsces)) % num_solnums ))
setidx=$(( (idx / (num_seeds * num_numsces * num_solnums)) % num_setnums )) 
methodidx=$(( (idx / (num_seeds * num_numsces * num_solnums * num_setnums)) % num_methods ))

method=${methods[$methodidx]}
solution_number=${solnums[$solidx]}
numsce=${numsces[$sceidx]}
seednum=${seeds[$seedidx]}
setnum=${setnums[$setidx]}

echo "Running with Task ID: $SLURM_ARRAY_TASK_ID"
echo "method=$method, solution_number=$solution_number, numsce=$numsce, seednum=$seednum, setnum=$setnum" 

python sol_validation.py \
    --method          "$method" \
    --solution_number "$solution_number" \
    --numsce          "$numsce" \
    --seednum         "$seednum" \
    --setnum          "$setnum" 