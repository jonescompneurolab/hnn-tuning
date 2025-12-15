#!/bin/bash
#SBATCH -J ERP_optim
#SBATCH -t 48:00:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=60

# Specify an output file
#SBATCH -o ERP_optim-%j.out
# also write errors to output
#SBATCH -e ERP_optim-%j.out

python optimize_drives_slurm.py --max_iter_time 200 --max_iter_drives 400 --n_procs 60 --scale_factor 1000 --smooth_window_len 5

# stats file
scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats