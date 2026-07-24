#!/bin/bash
#SBATCH -J opt_cma
#SBATCH -t 30:00
#SBATCH --mem=200G
#SBATCH --ntasks=10
#SBATCH --account carney-srj3-condo2
#SBATCH --exclude=node2430,node2414,node2428,node2418,node2416,node1940

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

# Step 2 of the ERP pipeline: CMA-ES optimisation of the evoked-drive parameters.
# 150 generations, population 100, spread over 100 MPI ranks.
python step2_optimize_ERP_cma.py --max_iter 2 --popsize 10 --n_procs 10 --scale_factor 1000 --smooth_window_len 5 --sigma0 0.25
# stats file
scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats