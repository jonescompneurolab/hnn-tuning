import os
import os.path as op
from functools import partial

import matplotlib.pyplot as plt

import hnn_core
from hnn_core.params import _short_name

from hnn_core import duecker_ET_model, simulate_dipole, read_dipole, MPIBackend
from hnn_core.optimization import Optimizer
import numpy as np

import pickle
hnn_core_root = op.join(op.dirname(hnn_core.__file__))

pathfig = os.path.join(os.getcwd(), 'figures')
#os.mkdir(pathfig)

def set_params(net, params):

    # Proximal 1
    synaptic_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_basket': 1., 'L5_pyramidal': 1.}
    
    weights_ampa_p1 = {'L2_basket':
                       params['prox1_ampa_L2_basket'],
                       'L2_pyramidal':
                       params['prox1_ampa_L2_pyramidal'],
                       'L5_basket':
                       params['prox1_ampa_L5_basket'],
                       'L5_pyramidal':
                       params['prox1_ampa_L5_pyramidal']}
    weights_nmda_p1 = {'L2_basket':
                       params['prox1_nmda_L2_basket'],
                       'L2_pyramidal':
                       params['prox1_nmda_L2_pyramidal'],
                       'L5_basket':
                       params['prox1_nmda_L5_basket'],
                       'L5_pyramidal':
                       params['prox1_nmda_L5_pyramidal']}
    
    net.add_evoked_drive('prox1',
                         mu=16.,
                         sigma=2.5,
                         numspikes=2,
                         location='proximal',
                         weights_ampa=weights_ampa_p1,
                         weights_nmda=weights_nmda_p1,
                         synaptic_delays=synaptic_delays_p, event_seed=544)

    # Distal
    synaptic_delays_d = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 1.}
    
    weights_ampa_d = {'L2_basket':
                       params['dist_ampa_L2_basket'],
                       'L2_pyramidal':
                       params['dist_ampa_L2_pyramidal'],
                       'L5_pyramidal':
                       params['dist_ampa_L5_pyramidal']}
    weights_nmda_d = {'L2_basket':
                       params['dist_nmda_L2_basket'],
                       'L2_pyramidal':
                       params['dist_nmda_L2_pyramidal'],
                       'L5_pyramidal':
                       params['dist_nmda_L5_pyramidal']}
    

    net.add_evoked_drive('dist',
                         mu=56.5,
                        sigma=2.5,
                        numspikes=3,
                         location='distal',
                         weights_ampa=weights_ampa_d,
                         weights_nmda=weights_nmda_d,
                         synaptic_delays=synaptic_delays_d, event_seed=544)


    weights_ampa_p2 = {'L2_basket':
                       params['prox2_ampa_L2_basket'],
                       'L2_pyramidal':
                       params['prox2_ampa_L2_pyramidal'],
                       'L5_basket':
                       params['prox2_ampa_L5_basket'],
                       'L5_pyramidal':
                       params['prox2_ampa_L5_pyramidal']}
    weights_nmda_p2 = {'L2_basket':
                       params['prox2_nmda_L2_basket'],
                       'L2_pyramidal':
                       params['prox2_nmda_L2_pyramidal'],
                       'L5_basket':
                       params['prox2_nmda_L5_basket'],
                       'L5_pyramidal':
                       params['prox2_nmda_L5_pyramidal']}
    
    net.add_evoked_drive('prox2',
                        mu=95,
                        sigma=10,
                        numspikes=3,
                        location='proximal',
                        weights_ampa=weights_ampa_p2,
                        weights_nmda=weights_nmda_p2,
                        synaptic_delays=synaptic_delays_p, event_seed=544)
    
    delay = net.delay

def optimize_drives(max_iter=1, n_procs=1, scale_factor=1000, smooth_window_len=5):

    data_url = '/oscar/home/kduecker/hnn-tuning/hnn-data/MEG_detection_data/S1_SupraT.txt'  # Added data loading
    exp_dpl = read_dipole(data_url)

    tstop = exp_dpl.times[-1]


    constraints = dict({'prox1_ampa_L2_basket': (0.01, 0.03),
                    'prox1_ampa_L2_pyramidal': (0.1, 0.3),
                    'prox1_ampa_L5_basket': (0.001, 0.1),
                    'prox1_ampa_L5_pyramidal': (0.001, 0.01),

                    'prox1_nmda_L2_basket': (0.0001, 0.0003),
                    'prox1_nmda_L2_pyramidal': (0.001, 0.1),
                    'prox1_nmda_L5_basket': (0.003, 0.005),
                    'prox1_nmda_L5_pyramidal': (0.001, 0.003),

                    'dist_ampa_L2_basket': (0.0001, 0.01),
                    'dist_ampa_L2_pyramidal': (0.01, .1),
                    'dist_ampa_L5_pyramidal': (0.05, 1.),

                    'dist_nmda_L2_basket': (0.0001, 0.0005),
                    'dist_nmda_L2_pyramidal': (0.0005, .1),
                    'dist_nmda_L5_pyramidal': (0.1, 1.),

                    'prox2_ampa_L2_basket': (0.0005, .01),
                    'prox2_ampa_L2_pyramidal': (0.001, 1.),
                    'prox2_ampa_L5_basket': (0.0001, .1),
                    'prox2_ampa_L5_pyramidal': (0.1, .5),

                    'prox2_nmda_L2_basket': (0.0001, 0.04),
                    'prox2_nmda_L2_pyramidal': (0.01, 0.5),
                    'prox2_nmda_L5_basket': (0.00005, 0.01),
                    'prox2_nmda_L5_pyramidal': (0.1, 1.)})

    net = duecker_2025_ET_model()

    optim = Optimizer(net, tstop=tstop, constraints=constraints,
                    set_params=set_params, solver='bayesian', obj_fun='dipole_rmse', max_iter=max_iter)

    with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'):
        optim.fit(target=exp_dpl, scale_factor=scale_factor, n_trials=1,
                smooth_window_len=smooth_window_len, bsl_cor='calcium')

    opt_dpl = simulate_dipole(optim.net_, tstop=tstop, n_trials=1, bsl_cor='calcium')[0]  # Added simulation
    opt_dpl.scale(scale_factor)
    opt_dpl.smooth(smooth_window_len)

    with open(f'optim_drives_{scale_factor}.pkl', 'wb') as file:
            pickle.dump(optim, file)
    with open(f'dpl_drives_{scale_factor}.pkl', 'wb') as file:
            pickle.dump(opt_dpl, file)
        
def set_params_time(scale_factor, net, params):

    synaptic_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}
    synaptic_delays_d = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 1.}


    with open(f'optim_drives_{scale_factor}.pkl', 'rb') as file:
        optim_drives = pickle.load(file)

    prox2_weights_ampa = optim_drives.net_.external_drives['prox2']['weights_ampa']
    prox2_weights_nmda = optim_drives.net_.external_drives['prox2']['weights_nmda']

    dist_weights_ampa = optim_drives.net_.external_drives['dist']['weights_ampa']
    dist_weights_nmda = optim_drives.net_.external_drives['dist']['weights_nmda']

    prox1_weights_ampa = optim_drives.net_.external_drives['prox1']['weights_ampa']
    prox1_weights_nmda = optim_drives.net_.external_drives['prox1']['weights_nmda']
    
    # Proximal 1
    
    net.add_evoked_drive('prox1',
                         mu = optim_drives.net_.external_drives['prox1']['dynamics']['mu'],
                         sigma = optim_drives.net_.external_drives['prox1']['dynamics']['sigma'],
                         numspikes = optim_drives.net_.external_drives['prox1']['dynamics']['numspikes'],
                         location='proximal',
                         weights_ampa=prox1_weights_ampa,
                         weights_nmda=prox1_weights_nmda,
                         synaptic_delays=synaptic_delays_p, event_seed=544)

    # Distal
    
    net.add_evoked_drive('dist',
                         mu=params['dist_mu'],
                         sigma=params['dist_sigma'],
                         numspikes=int(np.round(params['dist_num_spikes'])),
                         location='distal',
                         weights_ampa=dist_weights_ampa,
                         weights_nmda=dist_weights_nmda,
                         synaptic_delays=synaptic_delays_d, event_seed=544)
    
    net.add_evoked_drive('prox2',
                        mu=params['prox2_mu'],
                        sigma=params['prox2_sigma'],
                        numspikes=int(np.round(params['prox2_num_spikes'])),
                        location='proximal',
                        weights_ampa=prox2_weights_ampa,
                        weights_nmda=prox2_weights_nmda,
                        synaptic_delays=synaptic_delays_p, event_seed=544)
    
    delay = net.delay



def optimize_time(max_iter=1, n_procs=1, scale_factor=1000, smooth_window_len=5):

    data_url = '/oscar/home/kduecker/hnn-tuning/hnn-data/MEG_detection_data/S1_SupraT.txt'
    exp_dpl = read_dipole(data_url)
    tstop = exp_dpl.times[-1]

    constraints = dict({'dist_mu': (57, 62),
                    'dist_sigma': (2, 3),
                    'dist_num_spikes': (1, 6),
                    'prox2_mu': (94, 120),
                    'prox2_sigma': (8, 12),
                    'prox2_num_spikes': (1, 13)})

    net = duecker_2025_ET_model()
    set_par = partial(set_params_time,scale_factor)
    optim_time = Optimizer(net, tstop=tstop, constraints=constraints,
                    set_params=set_par, solver='bayesian', obj_fun='dipole_rmse', max_iter=max_iter)
    with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'):
        optim_time.fit(target=exp_dpl, scale_factor=scale_factor, n_trials=1,
                smooth_window_len=smooth_window_len, bsl_cor='calcium')

    with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'):
        opt_dpl = simulate_dipole(optim_time.net_, tstop=tstop, n_trials=1, bsl_cor='calcium')[0]

    opt_dpl.scale(scale_factor)
    opt_dpl.smooth(smooth_window_len)

    with open(f'optim_time_scale_{scale_factor}.pkl', 'wb') as file:
        pickle.dump(optim_time, file)
    with open(f'dpl_time_{scale_factor}.pkl', 'wb') as file:
        pickle.dump(opt_dpl, file)


def optimize_run(max_iter_time=1, max_iter_drives=1, n_procs=20, scale_factor=1000, smooth_window_len=5):
    
    optimize_drives(max_iter=max_iter_drives, n_procs=n_procs, scale_factor=scale_factor, smooth_window_len=smooth_window_len)  # Fixed typo

    optimize_time(max_iter=max_iter_time, n_procs=n_procs, scale_factor=scale_factor, smooth_window_len=smooth_window_len)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the optimization pipeline.')
    parser.add_argument('--max_iter_time', type=int, default=1,
                        help='Maximum iterations for time optimization')
    parser.add_argument('--max_iter_drives', type=int, default=1,
                        help='Maximum iterations for drives optimization')
    parser.add_argument('--n_procs', type=int, default=20,
                        help='Number of processors to use with MPI')
    parser.add_argument('--scale_factor', type=int, default=1000,
                        help='scale factor for dipole')
    parser.add_argument('--smooth_window_len', type=int, default=5,
                        help='scale factor for dipole')
    args = parser.parse_args()
    optimize_run(max_iter_time=args.max_iter_time,
                max_iter_drives=args.max_iter_drives,
                n_procs=args.n_procs, 
                scale_factor=args.scale_factor,
                smooth_window_len=args.smooth_window_len)

