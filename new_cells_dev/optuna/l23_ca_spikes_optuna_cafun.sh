#!/bin/bash
#SBATCH -J l23_optuna
#SBATCH -t 20:00
#SBATCH --mem=300G
#SBATCH --ntasks=10

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o l23_optuna-%j.out
#SBATCH -e l23_optuna-%j.out

python l23_ca_spikes_optuna.py --processes 10 --sampler random --round 1 --cafun

scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats