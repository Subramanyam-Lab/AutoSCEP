#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --time=12:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=log_valid/ph_validation-%A_%a.out
#SBATCH --error=log_valid/ph_validation-%A_%a.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=1-10000

source ~/.bashrc
conda activate neurmhsp
module load gurobi/10.0.3
export PYTHONPATH="${PYTHONPATH}:$(dirname $0)/../src"

readonly METHOD="PH"
readonly SOL_TIME=21600

readonly SOLNUMS=(11 12 13 14 15 16 17 18 19 20) 
readonly NUMSCES=(5 10 20) 
readonly SEEDS=({1..1000})
readonly SETNUMS=(1) 

num_solnums=${#SOLNUMS[@]}
num_numsces=${#NUMSCES[@]}
num_seeds=${#SEEDS[@]}
num_setnums=${#SETNUMS[@]}

idx=$(( SLURM_ARRAY_TASK_ID - 1 ))

seedidx=$(( idx % num_seeds ))
sceidx=$(( (idx / num_seeds) % num_numsces ))
solidx=$(( (idx / (num_seeds * num_numsces)) % num_solnums ))
setidx=$(( (idx / (num_seeds * num_numsces * num_solnums)) % num_setnums ))

solution_number=${SOLNUMS[$solidx]}
numsce=${NUMSCES[$sceidx]}
seednum=${SEEDS[$seedidx]}
setnum=${SETNUMS[$setidx]}


echo "Running with Task ID: $SLURM_ARRAY_TASK_ID"
echo "→ method:          $METHOD"
echo "→ solution_number: $solution_number"
echo "→ numsce:          $numsce"
echo "→ seednum:         $seednum"
echo "→ setnum:          $setnum"
echo "→ soltime:         $SOL_TIME"
echo "-----------------------------------"

python sol_validation.py \
    --method          "$METHOD" \
    --solution_number "$solution_number" \
    --numsce          "$numsce" \
    --seednum         "$seednum" \
    --setnum          "$setnum" \
    --soltime         "$SOL_TIME"