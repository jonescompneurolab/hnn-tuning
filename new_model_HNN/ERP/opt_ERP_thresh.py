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
pathgen = '/users/kduecker/hnn-tuning/hnn-data/'

from opt_ERP_cma import set_up_drives_empty


def get_constraints_init_params():

    # load the supra params as initial params
    with open('erp_supra_pred_params_cma.pkl', 'rb') as file:
        initial_params = pickle.load(file)


    constraints = dict({'prox1_L2_basket_ampa': (-15, -1.8),
                        'prox1_L2_pyramidal_ampa': (-2, 0),
                        'prox1_L5_basket_ampa': (-15, -1.75),
                        'prox1_L5ET_ampa': (-2, 0),

                        'prox1_L2_basket_nmda': (-15, -2.),
                        'prox1_L2_pyramidal_nmda': (-2, 0),
                        'prox1_L5_basket_nmda': (-15, -1.75),
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
                        'prox1_sigma': (1, 4),
                        'prox1_numspikes': (0, 3),
                        'dist1_mu': (50, 100),
                        'dist1_sigma': (1, 6),
                        'dist1_numspikes': (0,3),
                        'prox2_mu': (80, 150),
                        'prox2_sigma': (1, 20),
                        'prox2_numspikes': (0, 3)})

    return constraints, initial_params


def opt_cma(max_iter, popsize, n_procs, scaling_factor, smooth_win, which_erp):

    print(f"optimizing {which_erp} ERP...")
    data_url = f'{pathgen}MEG_detection_data/{which_erp}_trial_S1_ERP_all_avg.txt'
    exp_dpl = read_dipole(data_url)

    tstop = exp_dpl.times[-1]

    # get netwprl
    net_base = duecker_ET_model()
    # set up empty drives
    set_up_drives_empty(net_base)
    dt = 0.05           # increase dt to speed up 
    net_base._verbose = False

    constraints, initial_params = get_constraints_init_params()

    optim = Optimizer(net_base, tstop=tstop, constraints=constraints, solver='cma',
                        set_params=set_params_opt_drives, initial_params=initial_params, max_iter=max_iter, obj_fun="dipole_rmse_corr")

    optim.fit(target=exp_dpl, n_trials=3, scale_factor=scaling_factor,
            smooth_window_len=smooth_win, dt=dt, popsize=popsize, pth_backup=f'/users/kduecker/hnn-tuning/new_model_HNN/ERP/{which_erp}', 
            bsl_cor="duecker", n_jobs=n_procs, sigma0=0.15)

    with open(f'opt_threshold_ERP_cma_{popsize}_{which_erp}_wider_sigma.pkl', 'wb') as file:
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
    args = parser.parse_args()

    opt_cma(max_iter=args.max_iter,
            popsize=args.popsize,
            n_procs=args.n_procs, 
            scaling_factor=args.scale_factor,
            smooth_win=args.smooth_window_len,
            which_erp=args.which_erp)