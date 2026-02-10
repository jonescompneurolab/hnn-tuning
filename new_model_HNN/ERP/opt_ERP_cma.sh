#!/bin/bash
#SBATCH -J opt_cma
#SBATCH -t 48:00:00
#SBATCH --mem=140G
#SBATCH --ntasks=40
#SBATCH --exclude=node2430,node2414,node2428,node2418,node2416

# Specify an output file
#SBATCH -o opt_cma-%j.out
# also write errors to output
#SBATCH -e opt_cma-%j.out

# load modules
module load hpcx-mpi/2.25.1s-le4f
module load miniforge3/25.3.0-3-a6hh
# does not initialize automatically, so do it here
source ~/.bashrc
source ~/oscar-install/activate-hnn-core-conda-custom-source-env.sh
# activate environment
conda activate hnn-core-env
conda env list             # just double checking if we're in the right place

python opt_ERP_cma.py --max_iter 200 --popsize 250 --n_procs 40 --scale_factor 1000 --smooth_window_len 5

# stats file
scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats