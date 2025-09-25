#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=log_valid/ph_valid/EF_validation-%A_%a.out
#SBATCH --error=log_valid/ph_valid/EF_validation-%A_%a.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=2801-3000%100

source ~/.bashrc
conda activate neurmhsp
module load gurobi/10.0.3

readonly METHOD="EF"
readonly SOL_TIME=3000

# 잡 배열에서 사용할 파라미터 목록
readonly SOLNUMS=(11 12 13 14 15 16 17 18 19 20)
readonly NUMSCES=(5 10 20)
readonly SEEDS=(11 12 13 14 15 16 17 18 19 20)
readonly SETNUMS=(1 2 3 4 5 6 7 8 9 10)


num_solnums=${#SOLNUMS[@]}
num_numsces=${#NUMSCES[@]}
num_seeds=${#SEEDS[@]}
num_setnums=${#SETNUMS[@]}

# 1부터 시작하는 Slurm Task ID를 0부터 시작하는 인덱스로 변환
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))

# 선형적인 Task ID를 각 파라미터 인덱스로 변환 (가장 안쪽 루프부터 계산)
seedidx=$(( idx % num_seeds ))
sceidx=$(( (idx / num_seeds) % num_numsces ))
solidx=$(( (idx / (num_seeds * num_numsces)) % num_solnums ))
setidx=$(( (idx / (num_seeds * num_numsces * num_solnums)) % num_setnums ))

# 현재 Task에 해당하는 파라미터 값 할당
solution_number=${SOLNUMS[$solidx]}
numsce=${NUMSCES[$sceidx]}
seednum=${SEEDS[$seedidx]}
setnum=${SETNUMS[$setidx]}


# --- 작업 실행 ---
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