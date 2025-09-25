#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=8G         # memory per cpu-core
#SBATCH --time=24:00:00
#SBATCH --job-name=neurumhsp
#SBATCH --output=log/empire-%j.out
#SBATCH --error=log/empire-%j.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

source ~/.bashrc

conda activate neurmhsp
module purge
module load gurobi/10.0.3
module load cuda/11.5.0   
module load gcc/13.2.0
module load openmpi/4.1.1-pmi2


export MPICH_ASYNC_PROGRESS=1
echo "Running on nodes: $SLURM_NODELIST"


RHO=1e+1

# mpirun -n $SLURM_NTASKS python main.py --numsce $SLURM_NTASKS --rho $RHO

srun python main.py --numsce $SLURM_NTASKS --rho $RHO
