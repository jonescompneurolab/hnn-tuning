import os
import os.path as op
from functools import partial

import matplotlib.pyplot as plt

import hnn_core
from hnn_core.params import _short_name

from hnn_core import duecker_ET_model, simulate_dipole, read_dipole, MPIBackend, average_dipoles
from hnn_core.optimization import Optimizer, add_opt_drives
from hnn_core.network import pick_connection

import numpy as np

import pickle
hnn_core_root = op.join(op.dirname(hnn_core.__file__))
from scipy.stats import pearsonr

pathfig = os.path.join(os.getcwd(), 'figures')
pathgen = '/users/kduecker/hnn-tuning/hnn-data/'

from opt_ERP_cma import set_up_drives_empty


def get_constraints_init_params(which_erp):

    if which_erp == 'yes':

        initial_params = dict({'prox1_L2_basket_ampa': np.log10(0.001),
                        'prox1_L2_pyramidal_ampa': np.log10(0.014),
                        'prox1_L5_basket_ampa': np.log10(0.04),
                        'prox1_L5ET_ampa': np.log10(0.01),

                        'prox1_L2_basket_nmda': np.log10(0.009),
                        'prox1_L2_pyramidal_nmda': np.log10(0.025),
                        'prox1_L5_basket_nmda': np.log10(0.01),
                        'prox1_L5ET_nmda': np.log10(0.056),

                        'dist1_L2_basket_ampa': np.log10(6e-5),
                        'dist1_L2_pyramidal_ampa': np.log10(0.00016),
                        'dist1_L5ET_ampa': np.log10(0.5),

                        'dist1_L2_basket_nmda': -15,
                        'dist1_L2_pyramidal_nmda': np.log10(0.0017),
                        'dist1_L5ET_nmda': np.log(0.12),

                        'prox2_L2_basket_ampa': np.log10(0.0537),
                        'prox2_L2_pyramidal_ampa': np.log10(9.4e-5),
                        'prox2_L5_basket_ampa': np.log10(0.368),
                        'prox2_L5ET_ampa': np.log10(0.4),

                        'prox2_L2_basket_nmda': np.log10(1.2e-8),
                        'prox2_L2_pyramidal_nmda': np.log10(2.5e-7),
                        'prox2_L5_basket_nmda': np.log10(0.282),
                        'prox2_L5ET_nmda':  np.log10(0.3),
                        
                        'prox1_mu': 15.2,
                        'prox1_sigma': 6,
                        'dist1_mu': 55.5,
                        'dist1_sigma': 6,
                        'prox2_mu': 120,
                        'prox2_sigma': 12})

    elif which_erp == 'no':

        initial_params = dict({'prox1_L2_basket_ampa': np.log10(0.001),
                        'prox1_L2_pyramidal_ampa': np.log10(0.014),
                        'prox1_L5_basket_ampa': np.log10(0.2),
                        'prox1_L5ET_ampa': np.log10(0.01),

                        'prox1_L2_basket_nmda': np.log10(0.009),
                        'prox1_L2_pyramidal_nmda': np.log10(0.025),
                        'prox1_L5_basket_nmda': np.log10(0.02),
                        'prox1_L5ET_nmda': np.log10(0.056),

                        'dist1_L2_basket_ampa': np.log10(6e-5),
                        'dist1_L2_pyramidal_ampa': np.log10(0.00016),
                        'dist1_L5ET_ampa': np.log10(0.5),

                        'dist1_L2_basket_nmda': -15,
                        'dist1_L2_pyramidal_nmda': np.log10(0.0017),
                        'dist1_L5ET_nmda': np.log(0.3),

                        'prox2_L2_basket_ampa': np.log10(0.005),
                        'prox2_L2_pyramidal_ampa': np.log10(9.4e-5),
                        'prox2_L5_basket_ampa': np.log10(0.368),
                        'prox2_L5ET_ampa': np.log10(0.2),

                        'prox2_L2_basket_nmda': np.log10(1.2e-8),
                        'prox2_L2_pyramidal_nmda': np.log10(2.5e-7),
                        'prox2_L5_basket_nmda': np.log10(0.282),
                        'prox2_L5ET_nmda':  np.log10(0.1),
                        
                        'prox1_mu': 15.2,
                        'prox1_sigma': 6,
                        'dist1_mu': 58,
                        'dist1_sigma': 7,
                        'prox2_mu': 125,
                        'prox2_sigma': 12})

    constraints = dict({'prox1_L2_basket_ampa': (-15, -1.),
                        'prox1_L2_pyramidal_ampa': (-2, 0),
                        'prox1_L5_basket_ampa': (-15, -0.6),
                        'prox1_L5ET_ampa': (-2, 0),

                        'prox1_L2_basket_nmda': (-15, -1.),
                        'prox1_L2_pyramidal_nmda': (-2, 0),
                        'prox1_L5_basket_nmda': (-15, -1),
                        'prox1_L5ET_nmda': (-2, 0),

                        'dist1_L2_basket_ampa': (-15, 0),
                        'dist1_L2_pyramidal_ampa': (-6, 0),
                        'dist1_L5ET_ampa': (-6, 0),

                        'dist1_L2_basket_nmda': (-15, 0),
                        'dist1_L2_pyramidal_nmda': (-6, 0),
                        'dist1_L5ET_nmda': (-6, 0),

                        'prox2_L2_basket_ampa': (-15, 0),
                        'prox2_L2_pyramidal_ampa': (-15, -1),
                        'prox2_L5_basket_ampa': (-15, 0),
                        'prox2_L5ET_ampa': (-15, 0),

                        'prox2_L2_basket_nmda': (-15, 0),
                        'prox2_L2_pyramidal_nmda': (-15, -1),
                        'prox2_L5_basket_nmda': (-15, 0),
                        'prox2_L5ET_nmda': (-15, 0),
                        
                        'prox1_mu': (15, 25),
                        'prox1_sigma': (4,8),
                        'dist1_mu': (50, 70),
                        'dist1_sigma': (2, 10),
                        'prox2_mu': (90, 150),
                        'prox2_sigma': (9, 16)})

    return constraints, initial_params

def set_params_opt_drives_set_numspikes(net, param_values):
    drive_names = list(net.external_drives.keys())
    for name in drive_names:
        target_cell_types = net.external_drives[name]['target_types']

        net.external_drives[name]['dynamics']['mu'] = param_values[f'{name}_mu']
        net.external_drives[name]['dynamics']['sigma'] = param_values[f'{name}_sigma']
        # number of spikes is 1
        # reinstate external drives to be able to fill in below
        for receptor in ['ampa', 'nmda']:
                        net.external_drives[name][f'weights_{receptor}'] = {
                        ct: 0.0 for ct in target_cell_types
                    }

        for cell_type in target_cell_types:
            for receptor in ['ampa', 'nmda']:
                
                conn_idx = pick_connection(net, src_gids=name, target_gids=cell_type, receptor=receptor)
                assert len(conn_idx) == 1
                
                # KD: I still think this isn't the user-friendliest way as this is pretty hidden. 
                # Will fix.
                net.connectivity[conn_idx[0]]['nc_dict']['A_weight'] = 10 ** param_values[f'{name}_{cell_type}_{receptor}']
                net.external_drives[name][f'weights_{receptor}'][cell_type] = 10 ** param_values[f'{name}_{cell_type}_{receptor}']


def opt_cma(max_iter, popsize, n_procs, scaling_factor, smooth_win, which_erp, init_random, obj_fun):

    if init_random:
        print(f"Starting optimization {which_erp} ERP, with {obj_fun} and random initialization")
    else:
        print(f"Starting optimization {which_erp} ERP, with {obj_fun} and hand-tuned initialization")
    data_url = f'{pathgen}MEG_detection_data/{which_erp}_trial_S1_ERP_all_avg.txt'
    exp_dpl = read_dipole(data_url)

    tstop = exp_dpl.times[-1]

    # get netwprl
    net_base = duecker_ET_model()
    # set up empty drives
    set_up_drives_empty(net_base)
    dt = 0.05           # increase dt to speed up 
    net_base._verbose = False

    constraints, initial_params = get_constraints_init_params(which_erp=which_erp)

    if init_random:
        initial_params = None
        pth_backup = f'/users/kduecker/hnn-tuning/new_model_HNN/ERP/{which_erp}/init_random'
        os.makedirs(pth_backup, exist_ok=True)
    else:
        pth_backup = f'/users/kduecker/hnn-tuning/new_model_HNN/ERP/{which_erp}'
        os.makedirs(pth_backup, exist_ok=True)

    optim = Optimizer(net_base, tstop=tstop, constraints=constraints, solver='cma',
                        set_params=set_params_opt_drives_set_numspikes, initial_params=initial_params, max_iter=max_iter, obj_fun=obj_fun)

    optim.fit(target=exp_dpl, n_trials=3, scale_factor=scaling_factor,
            smooth_window_len=smooth_win, dt=dt, popsize=popsize, pth_backup=pth_backup, 
            bsl_cor="duecker", n_jobs=n_procs, sigma0=0.15)

    if init_random:
        filename=f'opt_threshold_ERP_cma_{popsize}_{which_erp}_wider_sigma_init_random_{obj_fun}.pkl'
    else:
        filename=f'opt_threshold_ERP_cma_{popsize}_{which_erp}_wider_sigma_{obj_fun}.pkl'
    
    with open(filename, 'wb') as file:
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
    parser.add_argument('--which_erp', type=str, default='yes',
                        help='which threshold ERP (yes or no)')
    parser.add_argument('--init_random', action='store_true', default=False,
                        help='Initiliaze random values?')
    parser.add_argument('--obj_fun', type=str, default='dipole_rmse_corr',
                        help="objective function")
    args = parser.parse_args()

    opt_cma(max_iter=args.max_iter,
            popsize=args.popsize,
            n_procs=args.n_procs, 
            scaling_factor=args.scale_factor,
            smooth_win=args.smooth_window_len,
            which_erp=args.which_erp,
            init_random=args.init_random,
            obj_fun=args.obj_fun)