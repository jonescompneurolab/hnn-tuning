"""Step 2 of the ERP tuning pipeline: optimise the evoked drives with CMA-ES.

Takes the hand-tuned drive parameters from Step 1 (`step1_handtune.ipynb`) as the
starting point and refines the evoked-drive timing and synaptic weights of the
`duecker_ET_model` against the measured S1 supra-threshold MEG ERP, using the
CMA-ES solver in hnn-core's `Optimizer`.

Meant to run on the cluster via `step2_optimize_ERP_cma.sh` (JOBLIB backend). The fitted
`Optimizer` is pickled to `opt_ERP_cma_<popsize>_sigma0_<sigma0>.pkl`.
"""

import os
import os.path as op
from functools import partial

import matplotlib.pyplot as plt

import hnn_core
from hnn_core.params import _short_name

from hnn_core import duecker_ET_model, simulate_dipole, read_dipole, MPIBackend, average_dipoles
from hnn_core.optimization import Optimizer
from hnn_core.network import pick_connection

import numpy as np

import pickle
hnn_core_root = op.join(op.dirname(hnn_core.__file__))
from scipy.stats import pearsonr

pathfig = os.path.join(os.getcwd(), 'figures')
from hnn_tuning import DATA, NETWORK_SIM
pathgen = str(DATA / 'hnn-data') + '/'

# path to back up cma runs
pth_backup=str(NETWORK_SIM / 'ERP' / 'backup_cma')
os.makedirs(pth_backup, exist_ok=True)

def set_params_opt_drives(net, param_values):
    """Write a flat CMA-ES parameter dict onto the network's evoked drives.

    This is the `set_params` callback handed to hnn-core's `Optimizer`: on every
    trial the solver proposes a `param_values`, including `mu`, `sigma`, `numspikes` and AMPA/NMDA weights.
    Weights are optimised in log10 space, so the stored value is
    `10 ** param_values[...]`
    """
    drive_names = list(net.external_drives.keys())
    for name in drive_names:
        target_cell_types = net.external_drives[name]['target_types']

        net.external_drives[name]['dynamics']['mu'] = param_values[f'{name}_mu']
        net.external_drives[name]['dynamics']['sigma'] = param_values[f'{name}_sigma']
        #import pdb; pdb.set_trace()
        net.external_drives[name]['dynamics']['numspikes'] = max(1, int(np.round(param_values[f'{name}_numspikes'])))

        # reinstate external drives to be able to fill in below
        for receptor in ['ampa', 'nmda']:
                        net.external_drives[name][f'weights_{receptor}'] = {
                        ct: 0.0 for ct in target_cell_types
                    }

        for cell_type in target_cell_types:
            for receptor in ['ampa', 'nmda']:
                
                conn_idx = pick_connection(net, src_gids=name, target_gids=cell_type, receptor=receptor)
                assert len(conn_idx) == 1
                net.connectivity[conn_idx[0]]['nc_dict']['A_weight'] = 10 ** param_values[f'{name}_{cell_type}_{receptor}']
                net.external_drives[name][f'weights_{receptor}'][cell_type] = 10 ** param_values[f'{name}_{cell_type}_{receptor}']

def get_constraints_init_params(continue_checkpoint=False):
    """Return the CMA-ES search bounds and the starting parameter values.

    `initial_params` are the hand-tuned values from Step 1 (log10 weights plus
    per-drive `mu`/`sigma`/`numspikes`); `constraints` gives the (low, high)
    bounds the solver stays within. If `continue_checkpoint`, the starting values
    are instead read from the best solution in a previous run's checkpoint
    (`backup_cma/cma_checkpoint.pkl`).
    """
    
    # log10 of the hand-tuned weights in step1_handtune.ipynb (0.0 -> -15, i.e. off)

    initial_params = dict({'prox1_L2_inhibitory_ampa': -2.4,    # 0.004 
                    'prox1_L2_pyramidal_ampa': -1.52,           # 0.03
                    'prox1_L5_inhibitory_ampa': -15,           # 0.0
                    'prox1_L5_pyramidal_ampa': -6.2,            # 6e-7

                    'prox1_L2_inhibitory_nmda': -2.,            # 0.01
                    'prox1_L2_pyramidal_nmda': -1.3,            # 0.05
                    'prox1_L5_inhibitory_nmda': -2,            # 0.01
                    'prox1_L5_pyramidal_nmda': -15,            # 0.0

                    'dist1_L2_inhibitory_ampa': -2.3,           # 0.005
                    'dist1_L2_pyramidal_ampa': -2.,             # 0.01
                    'dist1_L5_pyramidal_ampa': 0.,              # 1.0

                    'dist1_L2_inhibitory_nmda': -15,            # 0.0
                    'dist1_L2_pyramidal_nmda': -2.,             # 0.01
                    'dist1_L5_pyramidal_nmda': 0.,              # 1.0

                    'prox2_L2_inhibitory_ampa': -15,            # 0
                    'prox2_L2_pyramidal_ampa': -0.52,           # 0.3
                    'prox2_L5_inhibitory_ampa': -1.35,            # 0.044
                    'prox2_L5_pyramidal_ampa': -1.3,           # 0.051

                    'prox2_L2_inhibitory_nmda': -.066,            # 0.86
                    'prox2_L2_pyramidal_nmda': -4,            # 0.0001
                    'prox2_L5_inhibitory_nmda': -6.3,            # 5e-7
                    'prox2_L5_pyramidal_nmda': -0.42,            # 0.38

                    'prox1_mu': 15,
                    'prox1_sigma': 3.5,
                    'prox1_numspikes': 1.,
                    'dist1_mu': 56,
                    'dist1_sigma': 4.5,
                    'dist1_numspikes': 2,
                    'prox2_mu': 99,
                    'prox2_sigma': 12,
                    'prox2_numspikes': 1})

    constraints = dict({'prox1_L2_inhibitory_ampa': (-4, -1.8),
                        'prox1_L2_pyramidal_ampa': (-1.75, 0),
                        'prox1_L5_inhibitory_ampa': (-15, -6),
                        'prox1_L5_pyramidal_ampa': (-12, -1),

                        'prox1_L2_inhibitory_nmda': (-4, -2),
                        'prox1_L2_pyramidal_nmda': (-1.6, 0),
                        'prox1_L5_inhibitory_nmda': (-4, -2),
                        'prox1_L5_pyramidal_nmda': (-15, -1),

                        'dist1_L2_inhibitory_ampa': (-4.6, 0),
                        'dist1_L2_pyramidal_ampa': (-4, 0),
                        'dist1_L5_pyramidal_ampa': (-4, 0),

                        'dist1_L2_inhibitory_nmda': (-15, -10),
                        'dist1_L2_pyramidal_nmda': (-6, 0),
                        'dist1_L5_pyramidal_nmda': (-6, 0),

                        'prox2_L2_inhibitory_ampa': (-15, -10),
                        'prox2_L2_pyramidal_ampa': (-1.5, 0),
                        'prox2_L5_inhibitory_ampa': (-3, 0),
                        'prox2_L5_pyramidal_ampa': (-3, 0),

                        'prox2_L2_inhibitory_nmda': (-1.2, 0),
                        'prox2_L2_pyramidal_nmda': (-6, -2),
                        'prox2_L5_inhibitory_nmda': (-12, -4),
                        'prox2_L5_pyramidal_nmda': (-2, 0),
                        
                        'prox1_mu': (10, 25),
                        'prox1_sigma': (2, 8),
                        'prox1_numspikes': (0, 3),
                        'dist1_mu': (30, 70),
                        'dist1_sigma': (2, 8),
                        'dist1_numspikes': (0,3),
                        'prox2_mu': (80, 120),
                        'prox2_sigma': (6, 16),
                        'prox2_numspikes': (0, 3)})

    # if optimization crashed and we want to continue from checkpoint
    if continue_checkpoint:
        with open(op.join(pth_backup, 'cma_checkpoint.pkl'), 'rb') as file:
            cma =  pickle.load(file)
        predicted_params = cma['es'].result.xbest
        params_cma = {name: predicted_params[idx] for idx, name in enumerate(initial_params.keys())}

        return constraints, params_cma
    else:
        return constraints, initial_params

def set_up_drives_empty(net, tstop=200, n_prox=2, n_dist=1):
    """Attach the evoked drives to `net` with zero weight and placeholder timing.

    Creates the drive scaffold the optimiser fills in: `n_prox` proximal and
    `n_dist` distal evoked drives (named `prox1..`, `dist1..`) with all synaptic
    weights 0 and `mu=0`. `set_params_opt_drives` then overwrites their timing and
    weights on every trial. Only the synaptic delays are set here (fixed, not
    optimised).
    """

    prox_cell_type = ['L5_pyramidal', 'L5_inhibitory', 'L2_pyramidal', 'L2_inhibitory']
    dist_cell_type = ['L5_pyramidal', 'L2_pyramidal', 'L2_inhibitory']

    prox_weights  = {cell_type: 0.0 for cell_type in prox_cell_type}
    dist_weights  = {cell_type: 0.0 for cell_type in dist_cell_type}

    prox_delays = {"L2_inhibitory": 0.1, "L2_pyramidal": 0.1, "L5_inhibitory": 1.0, "L5_pyramidal": 1.0}
    dist_delays = {"L2_inhibitory": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}

    # Add proximal drives
    for idx in range(n_prox):
        name = f'prox{idx+1}'

        net.add_evoked_drive(name,
                             mu=0.0,
                             sigma=1.0,
                             numspikes=1,
                             location='proximal',
                             weights_ampa=prox_weights,
                             weights_nmda=prox_weights,
                             synaptic_delays=prox_delays)

    # Add distal drives
    for idx in range(n_dist):
        name = f'dist{idx+1}'

        net.add_evoked_drive(name,
                             mu=0.0,
                             sigma=1.0,
                             numspikes=1,
                             location='distal',
                             weights_ampa=dist_weights,
                             weights_nmda=dist_weights,
                             synaptic_delays=dist_delays)

def opt_cma(max_iter, popsize, n_procs, scaling_factor, smooth_win, sigma0, continue_checkpoint, init_random):
    """Run the CMA-ES optimisation and pickle the fitted `Optimizer`.

    Loads the target ERP, builds `duecker_ET_model` with empty drives, sets up the
    constraints/initial values, fits with CMA-ES (objective: dipole RMSE +
    anti-correlation, initial step size `sigma0`), and pickles the result to
    `opt_ERP_cma_<popsize>_sigma0_<sigma0>_dt0025.pkl` (or `..._random.pkl`).
    """

    if continue_checkpoint:
        print('Continuing optimization from checkpoint ...', flush=True)
    else:
        print('Starting new optimization using CMA')

    data_url = f'{pathgen}MEG_detection_data/S1_SupraT.txt'
    exp_dpl = read_dipole(data_url)

    tstop = exp_dpl.times[-1]

    # build the network and attach the (empty) drives the optimiser will fill in
    net_base = duecker_ET_model()
    set_up_drives_empty(net_base)
    dt = 0.025

    constraints, initial_params = get_constraints_init_params(continue_checkpoint)
    
    # Run optimization
    if init_random:
        initial_params = None
        print('random initialization!!!')
    

    optim = Optimizer(net_base, tstop=tstop, constraints=constraints, solver='cma', 
        set_params=set_params_opt_drives, initial_params=initial_params, max_iter=max_iter, obj_fun="dipole_rmse_corr")

    optim.fit(target=exp_dpl, n_trials=3, scale_factor=scaling_factor, sigma0=sigma0,
            smooth_window_len=smooth_win, dt=dt, popsize=popsize, pth_backup=pth_backup,
            bsl_cor="duecker", n_jobs=n_procs, verbose=False)

    # Drop the set_params callback before pickling to avoid set_params_opt_drives being a dependency. 
    optim._set_params = None

    if init_random:
        with open(f'opt_ERP_cma_{popsize}_sigma0_{sigma0}_random.pkl', 'wb') as file:
                    pickle.dump(optim, file)
    else:
        with open(f'opt_ERP_cma_{popsize}_sigma0_{sigma0}.pkl', 'wb') as file:
            pickle.dump(optim, file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the optimization pipeline.')
    parser.add_argument('--max_iter', type=int, default=1,
                        help='Maximum iterations')
    parser.add_argument('--popsize', type=int, default=1,
                        help='Population size')
    parser.add_argument('--n_procs', type=int, default=20,
                        help='Number of processors to use with Joblib (number of trials running in parallel).')
    parser.add_argument('--scale_factor', type=int, default=1000,
                        help='scale factor for dipole')
    parser.add_argument('--smooth_window_len', type=int, default=5,
                        help='scale factor for dipole')
    parser.add_argument('--sigma0', type=float, default=0.25,
                        help='CMA-ES initial step size (sigma0); baked into the output filename')
    parser.add_argument('--continue_checkpoint', action='store_true', default=False,
                        help='continue from checkpoint?')
    parser.add_argument('--init_random', action='store_true', default=False,
                        help='Initiliaze random values?')
    args = parser.parse_args()

    opt_cma(max_iter=args.max_iter,
            popsize=args.popsize,
            n_procs=args.n_procs, 
            scaling_factor=args.scale_factor,
            smooth_win=args.smooth_window_len,
            sigma0=args.sigma0,
            continue_checkpoint=args.continue_checkpoint,
            init_random=args.init_random)