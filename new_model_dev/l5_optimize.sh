#!/bin/bash
#SBATCH -J l5_optim
#SBATCH -t 100:00:00
#SBATCH --mem=80G

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o l5_optim-%j.out
#SBATCH -e l5_optim-%j.out

python l5_optimize_slurm.py

scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
