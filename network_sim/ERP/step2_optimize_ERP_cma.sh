#!/bin/bash
#SBATCH -J opt_cma
#SBATCH -t 10:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=50
#SBATCH --account carney-srj3-condo2

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

# Step 2 of the ERP pipeline: CMA-ES optimization of the evoked-drive parameters.
# 150 generations, population 100, distributedn over 100 jobs
python step2_optimize_ERP_cma.py --max_iter 50 --popsize 50 --n_procs 50 --scale_factor 1000 --smooth_window_len 5 --sigma0 0.05
# stats file
scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats