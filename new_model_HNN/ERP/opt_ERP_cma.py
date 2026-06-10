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
pathgen = '/users/kduecker/hnn-tuning/hnn-data/'


def add_opt_drives(net, tstop=200, n_prox=2, n_dist=1):
    prox_cell_type = ['L5_pyramidal', 'L5_inhibitory', 'L2_pyramidal', 'L2_inhibitory']
    dist_cell_type = ['L5_pyramidal', 'L2_pyramidal', 'L2_inhibitory']
    default_range = {'mu': (0.01, tstop), 'sigma': (0.01, 20),'numspikes':(0,1), 'ampa': (-5, 1), 'nmda': (-5, 1)}
    default_values = {'mu': tstop // 2, 'sigma': 2, 'ampa': -3, 'nmda': -3}

    prox_weights  = {cell_type: 0.0 for cell_type in prox_cell_type}
    dist_weights  = {cell_type: 0.0 for cell_type in dist_cell_type}

    prox_delays = {"L2_inhibitory": 0.1, "L2_pyramidal": 0.1, "L5_inhibitory": 1.0, "L5_pyramidal": 1.0,}
    dist_delays = {"L2_inhibitory": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}

    constraints, initial_params = dict(), dict()
    # Add proximal drives
    for idx in range(n_prox):
        name = f'evprox{idx+1}'
        constraints[f'{name}_mu'] = default_range['mu']
        initial_params[f'{name}_mu'] = np.random.uniform(*default_range['mu'])

        constraints[f'{name}_sigma'] = default_range['sigma']
        initial_params[f'{name}_sigma'] = np.random.uniform(*default_range['sigma'])

        constraints[f'{name}_numspikes'] = default_range['numspikes']
        initial_params[f'{name}_numspikes'] = 1

        for cell_type in prox_cell_type:
            constraints[f'{name}_{cell_type}_ampa'] = default_range['ampa']
            initial_params[f'{name}_{cell_type}_ampa'] = np.random.uniform(*default_range['ampa'])

            constraints[f'{name}_{cell_type}_nmda'] = default_range['nmda']
            initial_params[f'{name}_{cell_type}_nmda'] = np.random.uniform(*default_range['nmda'])

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
        name = f'evdist{idx+1}'
        constraints[f'{name}_mu'] = default_range['mu']
        initial_params[f'{name}_mu'] = np.random.uniform(*default_range['mu'])

        constraints[f'{name}_sigma'] = default_range['sigma']
        initial_params[f'{name}_sigma'] = np.random.uniform(*default_range['sigma'])

        constraints[f'{name}_numspikes'] = default_range['numspikes']
        initial_params[f'{name}_numspikes'] = 1

        for cell_type in prox_cell_type:
            constraints[f'{name}_{cell_type}_ampa'] = default_range['ampa']
            initial_params[f'{name}_{cell_type}_ampa'] = np.random.uniform(*default_range['ampa'])

            constraints[f'{name}_{cell_type}_nmda'] = default_range['nmda']
            initial_params[f'{name}_{cell_type}_nmda'] = np.random.uniform(*default_range['nmda'])

        net.add_evoked_drive(name,
                             mu=0.0,
                             sigma=1.0,
                             numspikes=1,
                             location='distal',
                             weights_ampa=dist_weights,
                             weights_nmda=dist_weights,
                             synaptic_delays=dist_delays)

    return constraints, initial_params

def set_params_opt_drives(net, param_values):
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
                
                # KD: I still think this isn't the user-friendliest way as this is pretty hidden. 
                # Will fix.
                net.connectivity[conn_idx[0]]['nc_dict']['A_weight'] = 10 ** param_values[f'{name}_{cell_type}_{receptor}']
                net.external_drives[name][f'weights_{receptor}'][cell_type] = 10 ** param_values[f'{name}_{cell_type}_{receptor}']

def get_constraints_init_params(continue_checkpoint=False):

    initial_params = dict({'prox1_L2_inhibitory_ampa': -1.8,
                    'prox1_L2_pyramidal_ampa': -1.85,
                    'prox1_L5_inhibitory_ampa': -15,
                    'prox1_L5_pyramidal_ampa': -2,

                    'prox1_L2_inhibitory_nmda': -2.4,
                    'prox1_L2_pyramidal_nmda': -1.6,
                    'prox1_L5_inhibitory_nmda': -12,
                    'prox1_L5_pyramidal_nmda': -1.3,

                    'dist1_L2_inhibitory_ampa': -3.5,
                    'dist1_L2_pyramidal_ampa': -4.2,
                    'dist1_L5_pyramidal_ampa': -0.0005,

                    'dist1_L2_inhibitory_nmda': -15,
                    'dist1_L2_pyramidal_nmda': -3,
                    'dist1_L5_pyramidal_nmda': -1,

                    'prox2_L2_inhibitory_ampa': -2,
                    'prox2_L2_pyramidal_ampa': -.5,
                    'prox2_L5_inhibitory_ampa': -3,
                    'prox2_L5_pyramidal_ampa': -0.5,

                    'prox2_L2_inhibitory_nmda': -2,
                    'prox2_L2_pyramidal_nmda': -.7,
                    'prox2_L5_inhibitory_nmda': -3,
                    'prox2_L5_pyramidal_nmda':  -0.7,
                    
                    'prox1_mu': 16.5,
                    'prox1_sigma': 3,
                    'prox1_numspikes': 1.,
                    'dist1_mu': 54,
                    'dist1_sigma': 4,
                    'dist1_numspikes': 2,
                    'prox2_mu': 112,
                    'prox2_sigma': 18.5,
                    'prox2_numspikes': 1})


    constraints = dict({'prox1_L2_inhibitory_ampa': (-15, -1.8),
                        'prox1_L2_pyramidal_ampa': (-1.85, 0),
                        'prox1_L5_inhibitory_ampa': (-15, -1.75),
                        'prox1_L5_pyramidal_ampa': (-2, 0),

                        'prox1_L2_inhibitory_nmda': (-15, -2.),
                        'prox1_L2_pyramidal_nmda': (-1.6, 0),
                        'prox1_L5_inhibitory_nmda': (-15, -1.75),
                        'prox1_L5_pyramidal_nmda': (-2, 0),

                        'dist1_L2_inhibitory_ampa': (-15, 0),
                        'dist1_L2_pyramidal_ampa': (-6, 0),
                        'dist1_L5_pyramidal_ampa': (-6, 0),

                        'dist1_L2_inhibitory_nmda': (-15, 0),
                        'dist1_L2_pyramidal_nmda': (-6, 0),
                        'dist1_L5_pyramidal_nmda': (-6, 0),

                        'prox2_L2_inhibitory_ampa': (-15, 0),
                        'prox2_L2_pyramidal_ampa': (-15, -.5),
                        'prox2_L5_inhibitory_ampa': (-15, 0),
                        'prox2_L5_pyramidal_ampa': (-15, -0.5),

                        'prox2_L2_inhibitory_nmda': (-15, 0),
                        'prox2_L2_pyramidal_nmda': (-15, -.5),
                        'prox2_L5_inhibitory_nmda': (-15, 0),
                        'prox2_L5_pyramidal_nmda': (-15, -0.5),
                        
                        'prox1_mu': (15, 25),
                        'prox1_sigma': (1, 4),
                        'prox1_numspikes': (0, 3),
                        'dist1_mu': (50, 100),
                        'dist1_sigma': (1, 6),
                        'dist1_numspikes': (0,3),
                        'prox2_mu': (80, 150),
                        'prox2_sigma': (1, 20),
                        'prox2_numspikes': (0, 3)})

    # if optimization crashed for some reason and we want to continue from checkpoint
    if continue_checkpoint:
        with open('cma_checkpoint.pkl', 'rb') as file:
            cma =  pickle.load(file)
        predicted_params = cma['es'].result.xbest
        params_cma = {name: predicted_params[idx] for idx, name in enumerate(initial_params.keys())}

        return constraints, params_cma
    else:
        return constraints, initial_params

def set_up_drives_empty(net, tstop=200, n_prox=2, n_dist=1):

    prox_cell_type = ['L5_pyramidal', 'L5_inhibitory', 'L2_pyramidal', 'L2_inhibitory']
    dist_cell_type = ['L5_pyramidal', 'L2_pyramidal', 'L2_inhibitory']
    default_range = {'mu': (0, tstop), 'sigma': (0, 20), 'ampa': (-5, 1), 'nmda': (-5, 1)}
    default_values = {'mu': tstop // 2, 'sigma': 2, 'ampa': -3, 'nmda': -3}

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

def opt_cma(max_iter, popsize, n_procs, scaling_factor, smooth_win, continue_checkpoint, init_random):


    if continue_checkpoint:
        print('Continuing optimization from checkpoint ...', flush=True)
    else:
        print('Starting new optimization using CMA')
        
    data_url = f'{pathgen}MEG_detection_data/S1_SupraT.txt'
    exp_dpl = read_dipole(data_url)

    tstop = exp_dpl.times[-1]

    # get netwprl
    net_base = duecker_ET_model()
    # set up empty drives
    set_up_drives_empty(net_base)
    dt = 0.05           # increase dt to speed up 

    constraints, initial_params = get_constraints_init_params(continue_checkpoint)
    
    # Run optimization
    if init_random:
        initial_params = None
        print('random initialization!!!')
    

    optim = Optimizer(net_base, tstop=tstop, constraints=constraints, solver='cma', 
        set_params=set_params_opt_drives, initial_params=initial_params, max_iter=max_iter, obj_fun="dipole_rmse_corr")

    optim.fit(target=exp_dpl, n_trials=3, scale_factor=scaling_factor, sigma0=0.25,
            smooth_window_len=smooth_win, dt=dt, popsize=popsize, pth_backup='/users/kduecker/hnn-tuning/new_model_HNN/ERP', 
            bsl_cor="duecker", n_jobs=n_procs, verbose=False)

    if init_random:
        with open(f'opt_ERP_cma_{popsize}_random.pkl', 'wb') as file:
                    pickle.dump(optim, file)
    else:
        with open(f'opt_ERP_cma_{popsize}.pkl', 'wb') as file:
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
            continue_checkpoint=args.continue_checkpoint,
            init_random=args.init_random)