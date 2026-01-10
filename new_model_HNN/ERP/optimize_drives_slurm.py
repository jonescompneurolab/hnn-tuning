import os
import os.path as op
from functools import partial

import matplotlib.pyplot as plt

import hnn_core
from hnn_core.params import _short_name

from hnn_core import duecker_ET_model, simulate_dipole, read_dipole, MPIBackend, average_dipoles
from hnn_core.optimization import Optimizer

import numpy as np

import pickle
hnn_core_root = op.join(op.dirname(hnn_core.__file__))
from scipy.stats import pearsonr

pathfig = os.path.join(os.getcwd(), 'figures')
pathgen = '/users/kduecker/hnn-data/'

def _rmse_corr(dpl, exp_dpl):

    """Calculates RMSE between data in dpl and exp_dpl
    weighted by correlation
    
    Parameters
    ----------
    dpl : instance of Dipole
        A dipole object with simulated data
    exp_dpl : instance of Dipole
        A dipole object with experimental data

    Returns
    -------
    err : float
        Weighted RMSE between data in dpl and exp_dpl
        rmse**(2-corr(dpl, exp_dpl))
    """
    from scipy import signal

    exp_start_index = exp_dpl.times.argmin()
    exp_end_index = exp_dpl.times.argmax()
    exp_length = exp_end_index - exp_start_index + 1

    sim_start_index = dpl.times.argmin()
    sim_end_index = dpl.times.argmax()
    sim_length = sim_end_index - sim_start_index + 1


    dpl1 = dpl.data["agg"]
    dpl2 = exp_dpl.data["agg"]

    if sim_length > exp_length:
        # downsample simulation timeseries to match exp data
        dpl1 = signal.resample(dpl1, exp_length)
    elif sim_length < exp_length:
        # downsample exp timeseries to match simulation data
        dpl2 = signal.resample(dpl2, sim_length)

    rmse = np.sqrt(((dpl1 - dpl2) ** 2).mean())
    sig_corr = pearsonr(dpl1, dpl2)[0]

    return rmse*(2-sig_corr)

def rmse_evk_corr(
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    obj_values,
    tstop,
    obj_fun_kwargs,
):
    """The objective function for evoked responses.

    Parameters
    ----------
    initial_net : instance of Network
        The network object.
    initial_params : dict
        Keys are parameter names, values are initial parameters.
    set_params : func
        User-defined function that sets network drives and parameters.
    predicted_params : list
        Parameters selected by the optimizer.
    update_params : func
        Function to update params.
    tstop : float
        The simulated dipole's duration.
    target : instance of Dipole
        A dipole object with experimental data.
    n_trials : int
        Number of trials to simulate and average.

    Returns
    -------
    obj : float
        Normalized RMSE between recorded and simulated dipole.
    """

    params = update_params(initial_params, predicted_params)

    # simulate dpl with predicted params
    new_net = initial_net.copy()
    set_params(new_net, params)

    if 'bsl_cor' in obj_fun_kwargs:
        dpls = simulate_dipole(new_net, tstop=tstop, n_trials=obj_fun_kwargs["n_trials"], bsl_cor=obj_fun_kwargs['bsl_cor'])
    else:
        dpls = simulate_dipole(new_net, tstop=tstop, n_trials=obj_fun_kwargs["n_trials"])

    # smooth & scale
    if "scale_factor" in obj_fun_kwargs:
        [dpl.scale(obj_fun_kwargs["scale_factor"]) for dpl in dpls]
    if "smooth_window_len" in obj_fun_kwargs:
        [dpl.smooth(obj_fun_kwargs["smooth_window_len"]) for dpl in dpls]

    dpl = average_dipoles(dpls)
    obj = _rmse_corr(dpl, obj_fun_kwargs["target"])
    obj_values.append(obj)

    return obj

def set_params_weights(net, params):

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
                         mu=17.5,
                         sigma=5.5,
                         numspikes=1,
                         location='proximal',
                         weights_ampa=weights_ampa_p1,
                         weights_nmda=weights_nmda_p1,
                         synaptic_delays=synaptic_delays_p)

    # Distal
    synaptic_delays_d = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1}
    
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
                         mu=53.5,
                        sigma=4.5,
                        numspikes=1,
                         location='distal',
                         weights_ampa=weights_ampa_d,
                         weights_nmda=weights_nmda_d,
                         synaptic_delays=synaptic_delays_d)


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
                        numspikes=1,
                        location='proximal',
                        weights_ampa=weights_ampa_p2,
                        weights_nmda=weights_nmda_p2,
                        synaptic_delays=synaptic_delays_p)
    
    delay = net.delay

def optimize_weights(max_iter=1, n_procs=1, scale_factor=1000, smooth_window_len=5):

    data_url = f'{pathgen}MEG_detection_data/S1_SupraT.txt'
    exp_dpl = read_dipole(data_url)

    tstop = exp_dpl.times[-1]


    constraints = dict({'prox1_ampa_L2_basket': (0.001, 0.01),
                    'prox1_ampa_L2_pyramidal': (0.075, 0.3),
                    'prox1_ampa_L5_basket': (0.00005, 0.0005),
                    'prox1_ampa_L5_pyramidal': (0.0005, 0.003),

                    'prox1_nmda_L2_basket': (0.001, 0.005),
                    'prox1_nmda_L2_pyramidal': (0.001, 0.01),
                    'prox1_nmda_L5_basket': (0.00001, 0.0001),
                    'prox1_nmda_L5_pyramidal': (0.0001, .001),

                    'dist_ampa_L2_basket': (0.001, 0.01),
                    'dist_ampa_L2_pyramidal': (0.01, .1),
                    'dist_ampa_L5_pyramidal': (0.1, 1.),

                    'dist_nmda_L2_basket': (0.00001, 0.002),
                    'dist_nmda_L2_pyramidal': (0.005, .04),
                    'dist_nmda_L5_pyramidal': (0.1, .6),

                    'prox2_ampa_L2_basket': (0.01, .2),
                    'prox2_ampa_L2_pyramidal': (0.1, 1.),
                    'prox2_ampa_L5_basket': (0.05, .5),
                    'prox2_ampa_L5_pyramidal': (0.1, 1.),

                    'prox2_nmda_L2_basket': (0.01, 0.5),
                    'prox2_nmda_L2_pyramidal': (0.01, 0.3),
                    'prox2_nmda_L5_basket': (0.01, 0.3),
                    'prox2_nmda_L5_pyramidal': (0.05, 0.9)})

    initial_params = dict({'prox1_ampa_L2_basket': 0.0025,
                    'prox1_ampa_L2_pyramidal': 0.125,
                    'prox1_ampa_L5_basket': 0.00015,
                    'prox1_ampa_L5_pyramidal': 0.0015,

                    'prox1_nmda_L2_basket': 0.0025,
                    'prox1_nmda_L2_pyramidal': 0.00175,
                    'prox1_nmda_L5_basket': 0.00005,
                    'prox1_nmda_L5_pyramidal': .0005,

                    'dist_ampa_L2_basket': 0.005,
                    'dist_ampa_L2_pyramidal': 0.015,
                    'dist_ampa_L5_pyramidal': 0.3,

                    'dist_nmda_L2_basket': 0.0005,
                    'dist_nmda_L2_pyramidal': 0.015,
                    'dist_nmda_L5_pyramidal': 0.2,

                    'prox2_ampa_L2_basket': 0.075,
                    'prox2_ampa_L2_pyramidal': 0.45,
                    'prox2_ampa_L5_basket': 0.1,
                    'prox2_ampa_L5_pyramidal': 0.75,

                    'prox2_nmda_L2_basket': 0.1,
                    'prox2_nmda_L2_pyramidal': 0.05,
                    'prox2_nmda_L5_basket': 0.15,
                    'prox2_nmda_L5_pyramidal':  0.55})

    net = duecker_ET_model()

    optim = Optimizer(net, tstop=tstop, constraints=constraints,
                    set_params=set_params_weights, solver='bayesian', obj_fun=rmse_evk_corr, max_iter=max_iter, initial_params=initial_params)

    with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'):
        optim.fit(target=exp_dpl, scale_factor=scale_factor, n_trials=5,
                smooth_window_len=smooth_window_len, bsl_cor='duecker')

    opt_dpl = simulate_dipole(optim.net_, tstop=tstop, n_trials=1, bsl_cor='duecker')[0]  # Added simulation
    opt_dpl.scale(scale_factor)
    opt_dpl.smooth(smooth_window_len)

    with open(f'optim_weights_{scale_factor}_nmdamodel.pkl', 'wb') as file:
            pickle.dump(optim, file)
    with open(f'dpl_weights_{scale_factor}_nmdamodel.pkl', 'wb') as file:
            pickle.dump(opt_dpl, file)
        
def set_params_time(scale_factor, net, params):

    synaptic_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}
    synaptic_delays_d = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1}


    with open(f'optim_weights_{scale_factor}_nmdamodel.pkl', 'rb') as file:
        optim_drives = pickle.load(file)

    prox2_weights_ampa = optim_drives.net_.external_drives['prox2']['weights_ampa']
    prox2_weights_nmda = optim_drives.net_.external_drives['prox2']['weights_nmda']

    dist_weights_ampa = optim_drives.net_.external_drives['dist']['weights_ampa']
    dist_weights_nmda = optim_drives.net_.external_drives['dist']['weights_nmda']

    prox1_weights_ampa = optim_drives.net_.external_drives['prox1']['weights_ampa']
    prox1_weights_nmda = optim_drives.net_.external_drives['prox1']['weights_nmda']
    
    # Proximal 1
    
    net.add_evoked_drive('prox1',
                         mu=params['prox1_mu'],
                         sigma=params['prox1_sigma'],
                         numspikes=int(np.round(params['prox1_num_spikes'])),
                         location='proximal',
                         weights_ampa=prox1_weights_ampa,
                         weights_nmda=prox1_weights_nmda,
                         synaptic_delays=synaptic_delays_p)

    # Distal
    
    net.add_evoked_drive('dist',
                         mu=params['dist_mu'],
                         sigma=params['dist_sigma'],
                         numspikes=int(np.round(params['dist_num_spikes'])),
                         location='distal',
                         weights_ampa=dist_weights_ampa,
                         weights_nmda=dist_weights_nmda,
                         synaptic_delays=synaptic_delays_d)
    
    net.add_evoked_drive('prox2',
                        mu=params['prox2_mu'],
                        sigma=params['prox2_sigma'],
                        numspikes=int(np.round(params['prox2_num_spikes'])),
                        location='proximal',
                        weights_ampa=prox2_weights_ampa,
                        weights_nmda=prox2_weights_nmda,
                        synaptic_delays=synaptic_delays_p)
    
    delay = net.delay



def optimize_time(max_iter=1, n_procs=1, scale_factor=1000, smooth_window_len=5):

    data_url = f'{pathgen}MEG_detection_data/S1_SupraT.txt'
    exp_dpl = read_dipole(data_url)
    tstop = exp_dpl.times[-1]

    constraints = dict({'prox1_mu': (17, 19.),
                    'prox1_sigma': (4., 6),
                    'prox1_num_spikes': (1, 1),
                    'dist_mu': (51, 68),
                    'dist_sigma': (3, 8),
                    'dist_num_spikes': (1, 2),
                    'prox2_mu': (90, 100),
                    'prox2_sigma': (5, 15),
                    'prox2_num_spikes': (1, 3)})

    initial_params = dict({'prox1_mu': 17.5,
                    'prox1_sigma': 5.5,
                    'prox1_num_spikes': 1,
                    'dist_mu': 53.5,
                    'dist_sigma': 4.5,
                    'dist_num_spikes': 1,
                    'prox2_mu': 95,
                    'prox2_sigma': 10,
                    'prox2_num_spikes': 1})

    net = duecker_ET_model()
    set_par = partial(set_params_time,scale_factor)
    optim_time = Optimizer(net, tstop=tstop, constraints=constraints,
                    set_params=set_par, solver='bayesian', obj_fun=rmse_evk_corr, max_iter=max_iter, initial_params=initial_params)
                    
    with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'):
        optim_time.fit(target=exp_dpl, scale_factor=scale_factor, n_trials=5,
                smooth_window_len=smooth_window_len, bsl_cor='duecker')

    with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'):
        opt_dpl = simulate_dipole(optim_time.net_, tstop=tstop, n_trials=1, bsl_cor='duecker')[0]

    opt_dpl.scale(scale_factor)
    opt_dpl.smooth(smooth_window_len)

    with open(f'optim_time_{scale_factor}_nmdamodel.pkl', 'wb') as file:
        pickle.dump(optim_time, file)
    with open(f'dpl_time_{scale_factor}_nmdamodel.pkl', 'wb') as file:
        pickle.dump(opt_dpl, file)


def optimize_run(max_iter_time=1, max_iter_weights=1, n_procs=20, scale_factor=1000, smooth_window_len=5):
    
    print("Starting optimization drive weights...")
    optimize_weights(max_iter=max_iter_weights, n_procs=n_procs, scale_factor=scale_factor, smooth_window_len=smooth_window_len)  # Fixed typo

    print("Starting optimization drive timing...")
    optimize_time(max_iter=max_iter_time, n_procs=n_procs, scale_factor=scale_factor, smooth_window_len=smooth_window_len)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the optimization pipeline.')
    parser.add_argument('--max_iter_time', type=int, default=1,
                        help='Maximum iterations for time optimization')
    parser.add_argument('--max_iter_weights', type=int, default=1,
                        help='Maximum iterations for synaptic weights optimization')
    parser.add_argument('--n_procs', type=int, default=20,
                        help='Number of processors to use with MPI')
    parser.add_argument('--scale_factor', type=int, default=1000,
                        help='scale factor for dipole')
    parser.add_argument('--smooth_window_len', type=int, default=5,
                        help='scale factor for dipole')
    args = parser.parse_args()
    optimize_run(max_iter_time=args.max_iter_time,
                max_iter_weights=args.max_iter_weights,
                n_procs=args.n_procs, 
                scale_factor=args.scale_factor,
                smooth_window_len=args.smooth_window_len)

