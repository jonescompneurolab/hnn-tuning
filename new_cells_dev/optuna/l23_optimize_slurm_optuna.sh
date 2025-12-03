#!/bin/bash
#SBATCH -J l23_optuna
#SBATCH -t 10:00:00
#SBATCH --mem=120G
#SBATCH --ntasks=33

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o l23_optuna-%j.out
#SBATCH -e l23_optuna-%j.out

python l23_optimize_slurm_optuna.py --processes 33 --sampler random

scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats