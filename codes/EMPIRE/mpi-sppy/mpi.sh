#!/bin/bash
#SBATCH --job-name=mpi4py-test   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=4               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=1G         # memory per cpu-core
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --output=log/empire-%j_%a.out
#SBATCH --error=log/empire-%j_%a.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=tzk5446@psu.edu

# source: https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py


source activate neurmhsp
module purge
module load cuda/11.5.0   
module load gcc/13.2.0
module load openmpi/4.1.1-pmi2

export MPICH_ASYNC_PROGRESS=1


# srun python test_mpi4py.py

mpirun -n 4 python test_mpi4py.py