import os
import os.path as op
from functools import partial

import matplotlib.pyplot as plt

import hnn_core
from hnn_core.params import _short_name

from hnn_core import duecker_ET_model, simulate_dipole, read_dipole, MPIBackend, average_dipoles
from hnn_core.optimization import Optimizer, add_opt_drives, set_params_opt_drives

import numpy as np

import pickle
hnn_core_root = op.join(op.dirname(hnn_core.__file__))
from scipy.stats import pearsonr

pathfig = os.path.join(os.getcwd(), 'figures')
pathgen = '/users/kduecker/hnn-data/'


def get_constraints_init_params():

    initial_params = dict({'prox1_L2_basket_ampa': np.log10(0.001),
                    'prox1_L2_pyramidal_ampa': np.log10(0.15),
                    'prox1_L5_basket_ampa': np.log10(1e-15),
                    'prox1_L5_pyramidal_ampa': np.log10(0.001),

                    'prox1_L2_basket_nmda': np.log10(0.01),
                    'prox1_L2_pyramidal_nmda': np.log10(0.05),
                    'prox1_L5_basket_nmda': np.log10(0.0075),
                    'prox1_L5_pyramidal_nmda': np.log10(1e-15),

                    'dist1_L2_basket_ampa': np.log10(0.005),
                    'dist1_L2_pyramidal_ampa': np.log10(1),
                    'dist1_L5_pyramidal_ampa': np.log10(0.8),

                    'dist1_L2_basket_nmda': np.log10(1e-15),
                    'dist1_L2_pyramidal_nmda': np.log10(0.5),
                    'dist1_L5_pyramidal_nmda': np.log10(0.1),

                    'prox2_L2_basket_ampa': np.log10(0.009),
                    'prox2_L2_pyramidal_ampa': np.log10(0.4),
                    'prox2_L5_basket_ampa': np.log10(0.006),
                    'prox2_L5_pyramidal_ampa': np.log10(0.2),

                    'prox2_L2_basket_nmda': np.log10(0.1),
                    'prox2_L2_pyramidal_nmda': np.log10(0.05),
                    'prox2_L5_basket_nmda': np.log10(0.005),
                    'prox2_L5_pyramidal_nmda':  np.log10(0.8),
                    
                    'prox1_mu': 20,
                    'prox1_sigma': 6,
                    'prox1_numspikes': 1,
                    'dist1_mu': 62,
                    'dist1_sigma': 7,
                    'dist1_numspikes': 2,
                    'prox2_mu': 95,
                    'prox2_sigma': 10,
                    'prox2_numspikes': 1})


    constraints = dict({'prox1_L2_basket_ampa': (-15, -1.75),
                        'prox1_L2_pyramidal_ampa': (-2, 0.25),
                        'prox1_L5_basket_ampa': (-15, -1.75),
                        'prox1_L5_pyramidal_ampa': (-15, 0),

                        'prox1_L2_basket_nmda': (-15, -1.75),
                        'prox1_L2_pyramidal_nmda': (-2, 0.25),
                        'prox1_L5_basket_nmda': (-15, -1.75),
                        'prox1_L5_pyramidal_nmda': (-15, 0),

                        'dist1_L2_basket_ampa': (-15, 0),
                        'dist1_L2_pyramidal_ampa': (-15, 0),
                        'dist1_L5_pyramidal_ampa': (-15, 0),

                        'dist1_L2_basket_nmda': (-15, 0),
                        'dist1_L2_pyramidal_nmda': (-15, 0),
                        'dist1_L5_pyramidal_nmda': (-15, 0),

                        'prox2_L2_basket_ampa': (-15, 0),
                        'prox2_L2_pyramidal_ampa': (-15, 0),
                        'prox2_L5_basket_ampa': (-15, 0),
                        'prox2_L5_pyramidal_ampa': (-15, 0),

                        'prox2_L2_basket_nmda': (-15, 0),
                        'prox2_L2_pyramidal_nmda': (-15, 0),
                        'prox2_L5_basket_nmda': (-15, 0),
                        'prox2_L5_pyramidal_nmda': (-15, 0),
                        
                        'prox1_mu': (5, 30),
                        'prox1_sigma': (0.1, 20),
                        'prox1_numspikes': (0, 3),
                        'dist1_mu': (50, 100),
                        'dist1_sigma': (0.1, 20),
                        'dist1_numspikes': (0,3),
                        'prox2_mu': (80, 150),
                        'prox2_sigma': (0.1, 20),
                        'prox2_numspikes': (0, 3)})
        
    return constraints, initial_params

def set_up_drives_empty(net, tstop=200, n_prox=2, n_dist=1):

    prox_cell_type = ['L5_pyramidal', 'L5_basket', 'L2_pyramidal', 'L2_basket']
    dist_cell_type = ['L5_pyramidal', 'L2_pyramidal', 'L2_basket']
    default_range = {'mu': (0, tstop), 'sigma': (0, 20), 'ampa': (-5, 1), 'nmda': (-5, 1)}
    default_values = {'mu': tstop // 2, 'sigma': 2, 'ampa': -3, 'nmda': -3}

    prox_weights  = {cell_type: 0.0 for cell_type in prox_cell_type}
    dist_weights  = {cell_type: 0.0 for cell_type in dist_cell_type}

    prox_delays = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_basket": 1.0, "L5_pyramidal": 1.0,}
    dist_delays = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}

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

def opt_cma(max_iter, popsize, n_procs, scaling_factor, smooth_win):
    data_url = f'{pathgen}MEG_detection_data/S1_SupraT.txt'
    exp_dpl = read_dipole(data_url)

    tstop = exp_dpl.times[-1]

    # get netwprl
    net_base = duecker_ET_model()
    # set up empty drives
    set_up_drives_empty(net_base)
    dt = 0.025           # increase dt to speed up 
    net_base._verbose = True

    constraints, initial_params = get_constraints_init_params()
    # Run optimization
    optim = Optimizer(net_base, tstop=tstop, constraints=constraints, solver='cma',
                    set_params=set_params_opt_drives, initial_params=initial_params, max_iter=max_iter, obj_fun="dipole_rmse_corr")


    with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'):        
        optim.fit(target=exp_dpl, n_trials=5, scale_factor=scaling_factor,
                smooth_window_len=smooth_win, dt=dt, popsize=popsize, bsl_cor="duecker", n_jobs=n_procs)

    with open(f'opt_ERP_cma_strict_constraints.pkl', 'wb') as file:
        pickle.dump(optim, file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the optimization pipeline.')
    parser.add_argument('--max_iter', type=int, default=1,
                        help='Maximum iterations')
    parser.add_argument('--popsize', type=int, default=1,
                        help='Population size')
    parser.add_argument('--n_procs', type=int, default=20,
                        help='Number of processors to use with MPI')
    parser.add_argument('--scale_factor', type=int, default=1000,
                        help='scale factor for dipole')
    parser.add_argument('--smooth_window_len', type=int, default=5,
                        help='scale factor for dipole')
    args = parser.parse_args()

    opt_cma(max_iter=args.max_iter,
                popsize=args.popsize,
                n_procs=args.n_procs, 
                scaling_factor=args.scale_factor,
                smooth_win=args.smooth_window_len)