""""Compute alpha-beta ratios for different parameters and save results to pickle files"""

import numpy as np
import pandas as pd
import os
import os.path as op
import argparse
from alpha_beta_complex import run_alpha_beta_complex_simulations, sim_analysis_params_default, alpha_beta_ratio

# custom scripts and paths
import sys
sys.path.append('/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/new_model_HNN/')
main_pth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/new_model_HNN/alpha_beta/'
out_pth = op.join(main_pth, 'sim_results')
os.makedirs(out_pth, exist_ok=True)

def delay_ratio(n_procs):
    """Compute alpha-beta ratio for different feedback delays"""
    delays = np.arange(0, 95, 5)
    ab_ratio = np.zeros_like(delays, dtype=float)
    for i, delay in enumerate(delays):
        params = sim_analysis_params_default()
        params['fb_delay'] = delay
        params['n_procs'] = n_procs
        dpl, _ = run_alpha_beta_complex_simulations(params)
        ratio, _, _ = alpha_beta_ratio(dpl)
        
        ab_ratio[i] = ratio.mean()

    df_delay = pd.DataFrame(data={'fb_delay': delays, 'ab_ratio': ab_ratio})
    df_delay.to_pickle(op.join(out_pth, 'ab_ratio_delay.pkl'))

def burst_std_ratio(n_procs):
    """Compute alpha-beta ratio for different proximal burst std"""
    burst_stds = np.arange(np.sqrt(200), np.sqrt(600), 4)
    burst_ratio_prox = np.zeros_like(burst_stds, dtype=float)
    burst_ratio_dist = np.zeros_like(burst_stds, dtype=float)

    for i, bstd in enumerate(burst_stds):
        params = sim_analysis_params_default()
        params['burst_std_prox'] = bstd
        params['n_procs'] = n_procs
        dpl, _ = run_alpha_beta_complex_simulations(params)
        ab_ratio, _, _ = alpha_beta_ratio(dpl).mean()
        burst_ratio_prox[i] = ab_ratio.mean()

    for i, bstd in enumerate(burst_stds):
        params = sim_analysis_params_default()
        params['burst_std_dist'] = bstd
        params['n_procs'] = n_procs
        dpl, _ = run_alpha_beta_complex_simulations(params)
        ab_ratio, _, _ = alpha_beta_ratio(dpl).mean()
        burst_ratio_dist[i] = ab_ratio.mean()

    df_burst = pd.DataFrame(data={'burst_std_prox': burst_stds, 'ab_ratio_prox': burst_ratio_prox, 'ab_ratio_dist': burst_ratio_dist})
    df_burst.to_pickle(op.join(out_pth, 'ab_ratio_burst.pkl'))

def ndrive_cells_ratio(n_procs):
    num_cells = np.arange(5, 15, 5)
    num_ratio_prox = np.zeros_like(num_cells, dtype=float)
    num_ratio_dist = np.zeros_like(num_cells, dtype=float)

    for n, n_cells in enumerate(num_cells):
        params = sim_analysis_params_default()
        params['n_drive_cells_prox'] = n_cells
        params['n_procs'] = n_procs
        dpl, _ = run_alpha_beta_complex_simulations(params)
        ab_ratio, _, _ = alpha_beta_ratio(dpl).mean()
        num_ratio_prox[n] = ab_ratio.mean()

    for n, n_cells in enumerate(num_cells):
        params = sim_analysis_params_default()
        params['n_drive_cells_dist'] = n_cells
        params['n_procs'] = n_procs
        dpl, _ = run_alpha_beta_complex_simulations(params)
        ab_ratio, _, _ = alpha_beta_ratio(dpl).mean()
        num_ratio_dist[n] = ab_ratio.mean()

    df_ncells = pd.DataFrame(data={'n_drive_cells': num_cells, 'ab_ratio_prox': num_ratio_prox, 'ab_ratio_dist': num_ratio_dist})
    df_ncells.to_pickle(op.join(out_pth, 'ab_ratio_ncells.pkl'))

def ampa_weights_ratio(n_procs):
    ampa_weights = np.linspace(0.00002, 0.00006, 3)
    ampa_ratio_prox = np.zeros_like(ampa_weights, dtype=float)
    ampa_ratio_dist = np.zeros_like(ampa_weights, dtype=float)

    for a, w in enumerate(ampa_weights):
        params = sim_analysis_params_default()
        params['weights_ampa_p'] = {'L2_basket': 0.0, 'L2_pyramidal': w, 'L5_pyramidal': w, 'L5_basket': 0.0} # proximal weights
        params['n_procs'] = n_procs
        dpl, _ = run_alpha_beta_complex_simulations(params)
        ab_ratio, _, _ = alpha_beta_ratio(dpl).mean()
        ampa_ratio_prox[a] = ab_ratio.mean()

    for a, w in enumerate(ampa_weights):
        params = sim_analysis_params_default()
        params['weights_ampa_d'] = {'L2_basket': 0.0, 'L2_pyramidal': w, 'L5_pyramidal': w, 'L5_basket': 0.0} # distal weights
        params['n_procs'] = n_procs
        dpl, _ = run_alpha_beta_complex_simulations(params)
        ab_ratio, _, _ = alpha_beta_ratio(dpl).mean()
        ampa_ratio_dist[a] = ab_ratio.mean()
    
    df_ampa = pd.DataFrame(data={'ampa_weights': ampa_weights, 'ab_ratio_prox': ampa_ratio_prox, 'ab_ratio_dist': ampa_ratio_dist})
    df_ampa.to_pickle(op.join(out_pth, 'ab_ratio_ampa.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alpha-beta complex, reproduce Jones et al., 2009.')

    parser.add_argument('--n_procs', type=int, default=1,
                        help='Number of cores for MPI simulation (default: 1, run serially)')
    args = parser.parse_args()

    n_procs = args.n_procs
    delay_ratio(n_procs)
    burst_std_ratio(n_procs)
    ndrive_cells_ratio(n_procs)
    ampa_weights_ratio(n_procs)