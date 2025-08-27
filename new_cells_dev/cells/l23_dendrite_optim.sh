#!/bin/bash
#SBATCH -J l23_dendrite_cmaes
#SBATCH -t 10:00:00
#SBATCH --mem=120G
#SBATCH --ntasks=30

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o l23_dendrite_cmaes-%j.out
#SBATCH -e l23_dendrite_cmaes-%j.out

python l23_dendrite_optim.py --processes 30 --sampler cmaes

scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats