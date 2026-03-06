#!/bin/bash
#SBATCH -J mu_sim
#SBATCH -t 24:00:00
#SBATCH --mem=80G
#SBATCH --ntasks=60
#SBATCH --exclude=node2430,node2414,node2428,node2418
#SBATCH --array 2-5

# Specify an output file
#SBATCH -o mu_sim-%j.out
# also write errors to output
#SBATCH -e mu_sim-%j.out

# load modules
module load hpcx-mpi/2.25.1s-le4f
module load miniforge3/25.3.0-3-a6hh
# does not initialize automatically, so do it here
source ~/.bashrc
source ~/oscar-install/activate-hnn-core-conda-custom-source-env.sh
# activate environment
conda activate hnn-core-env
conda env list             # just double checking if we're in the right place

python alpha_beta_complex.py --n_trials 600 --n_procs 60 --run_idx ${SLURM_ARRAY_TASK_ID}

# stats file
scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats