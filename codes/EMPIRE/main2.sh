#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100GB 
#SBATCH --time=6:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=empire_log5/empire-%A_%a.out
#SBATCH --error=empire_log5/empire-%A_%a.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --array=783-800%10

source ~/.bashrc
source activate neurmhsp
module load gurobi/10.0.3

# azs7266_sc
# parameter 리스트
LENREG_LIST=(1 2 4 8 16 32 64 128)
SOLNUM_LIST=(1 2 3 4 5 6 7 8 9 10)

# 0-based INDEX
INDEX=$(( SLURM_ARRAY_TASK_ID - 1 ))

# solnum 인덱스: INDEX를 (8×10)=80으로 나눈 몫
SOLNUM=${SOLNUM_LIST[$(( INDEX / 80 ))]}

# 나머지(rem)를 이용해 lenreg, seed 계산
rem=$(( INDEX % 80 ))
lenreg=${LENREG_LIST[$(( rem / 10 ))]}
seed=$(( rem % 10 + 1 ))

echo "Job $SLURM_ARRAY_TASK_ID → solnum=$SOLNUM, lenreg=$lenreg, seed=$seed"



# /storage/home/tzk5446/.conda/envs/neurmhsp/bin/python REDUCED_NEUREMPIRE.py --seed $seed --lenreg $lenreg
/storage/home/tzk5446/.conda/envs/neurmhsp/bin/python empire_model_sol_validation.py --seed $seed --solution_number $SOLNUM --lenregseason $lenreg


