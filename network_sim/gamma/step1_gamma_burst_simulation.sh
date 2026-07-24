#!/bin/bash
#SBATCH -J gamma_bursty
#SBATCH -t 4:00:00
#SBATCH --mem=80G
#SBATCH --ntasks=50
#SBATCH --account carney-srj3-condo2
#SBATCH --exclude=node2430,node2414,node2428,node2418

# Specify an output file
#SBATCH -o gamma_bursty-%j.out
# also write errors to output
#SBATCH -e gamma_bursty-%j.out

# load modules
module load hpcx-mpi/2.25.1s-le4f
module load miniforge3/25.3.0-3-a6hh
# does not initialize automatically, so do it here
source ~/.bashrc
source ~/oscar-install/activate-hnn-core-conda-custom-source-env.sh
# activate environment
conda activate hnn-core-env
conda env list             # just double checking if we're in the right place

# Bursty gamma (manuscript Fig. 11h-k): 100 trials over 50 Joblib workers.
# Writes net_gamma_bursty.pkl / dpl_gamma_bursty.pkl to sim_results/,
# which step2_inspect_gamma_multiscale.ipynb loads and plots.
python step1_gamma_burst_simulation.py --n_trials 100 --n_procs 50

# stats file
scontrol show job $SLURM_JOB_ID >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
