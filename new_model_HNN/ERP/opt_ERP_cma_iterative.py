"""Iterative (drive-by-drive) CMA-ES optimization of an evoked response.

Instead of optimizing all drives simultaneously over the full simulation, this
script tunes one drive at a time while holding the others fixed, and restricts
the objective's comparison window to the interval associated with that drive:

    prox1 :  0 - 45 ms
    dist1 : 45 - 90 ms
    prox2 : 90 - end of simulation

This mirrors the "chunked" protocol in hnn_core's legacy ``optimize_evoked``
(weighted-RMSE per time window), but uses CMA-ES and the dipole_rmse_corr
objective. Each chunk optimizes ~9-11 parameters instead of ~29, which is far
better conditioned for CMA.
"""

import os
import os.path as op
from collections import OrderedDict

import numpy as np
import pickle

import hnn_core
from hnn_core import duecker_ET_model, simulate_dipole, read_dipole, average_dipoles
from hnn_core.optimization import Optimizer
from hnn_core.network import pick_connection

# private helpers reused from the stock objective machinery
from hnn_core.optimization.objective_functions import _preprocess_dipole
from hnn_core.batch_simulate import BatchSimulate
from hnn_core.dipole import _rmse_corr

hnn_core_root = op.join(op.dirname(hnn_core.__file__))

pathfig = os.path.join(os.getcwd(), 'figures')
pathgen = '/users/kduecker/hnn-tuning/hnn-data/'
pth_backup = '/users/kduecker/hnn-tuning/new_model_HNN/ERP'


# ---------------------------------------------------------------------------
# Drive setup / parameter mapping (unchanged from opt_ERP_cma.py)
# ---------------------------------------------------------------------------
def set_up_drives_empty(net, tstop=200, n_prox=2, n_dist=1):
    prox_cell_type = ['L5_pyramidal', 'L5_inhibitory', 'L2_pyramidal', 'L2_inhibitory']
    dist_cell_type = ['L5_pyramidal', 'L2_pyramidal', 'L2_inhibitory']

    prox_weights = {cell_type: 0.0 for cell_type in prox_cell_type}
    dist_weights = {cell_type: 0.0 for cell_type in dist_cell_type}

    prox_delays = {"L2_inhibitory": 0.1, "L2_pyramidal": 0.1, "L5_inhibitory": 1.0, "L5_pyramidal": 1.0}
    dist_delays = {"L2_inhibitory": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}

    for idx in range(n_prox):
        name = f'prox{idx + 1}'
        net.add_evoked_drive(name, mu=0.0, sigma=1.0, numspikes=1,
                             location='proximal', weights_ampa=prox_weights,
                             weights_nmda=prox_weights, synaptic_delays=prox_delays)

    for idx in range(n_dist):
        name = f'dist{idx + 1}'
        net.add_evoked_drive(name, mu=0.0, sigma=1.0, numspikes=1,
                             location='distal', weights_ampa=dist_weights,
                             weights_nmda=dist_weights, synaptic_delays=dist_delays)


def set_params_opt_drives(net, param_values):
    drive_names = list(net.external_drives.keys())
    for name in drive_names:
        target_cell_types = net.external_drives[name]['target_types']

        net.external_drives[name]['dynamics']['mu'] = param_values[f'{name}_mu']
        net.external_drives[name]['dynamics']['sigma'] = param_values[f'{name}_sigma']
        net.external_drives[name]['dynamics']['numspikes'] = max(
            1, int(np.round(param_values[f'{name}_numspikes'])))

        for receptor in ['ampa', 'nmda']:
            net.external_drives[name][f'weights_{receptor}'] = {
                ct: 0.0 for ct in target_cell_types}

        for cell_type in target_cell_types:
            for receptor in ['ampa', 'nmda']:
                conn_idx = pick_connection(net, src_gids=name,
                                           target_gids=cell_type, receptor=receptor)
                assert len(conn_idx) == 1
                weight = 10 ** param_values[f'{name}_{cell_type}_{receptor}']
                net.connectivity[conn_idx[0]]['nc_dict']['A_weight'] = weight
                net.external_drives[name][f'weights_{receptor}'][cell_type] = weight


def get_constraints_init_params():
    initial_params = dict({
        'prox1_L2_inhibitory_ampa': -1.8, 'prox1_L2_pyramidal_ampa': -1.85,
        'prox1_L5_inhibitory_ampa': -15, 'prox1_L5_pyramidal_ampa': -2,
        'prox1_L2_inhibitory_nmda': -2.4, 'prox1_L2_pyramidal_nmda': -1.6,
        'prox1_L5_inhibitory_nmda': -12, 'prox1_L5_pyramidal_nmda': -1.3,
        'dist1_L2_inhibitory_ampa': -3.5, 'dist1_L2_pyramidal_ampa': -4.2,
        'dist1_L5_pyramidal_ampa': -0.0005,
        'dist1_L2_inhibitory_nmda': -15, 'dist1_L2_pyramidal_nmda': -3,
        'dist1_L5_pyramidal_nmda': -1,
        'prox2_L2_inhibitory_ampa': -2, 'prox2_L2_pyramidal_ampa': -.5,
        'prox2_L5_inhibitory_ampa': -3, 'prox2_L5_pyramidal_ampa': -0.5,
        'prox2_L2_inhibitory_nmda': -2, 'prox2_L2_pyramidal_nmda': -.7,
        'prox2_L5_inhibitory_nmda': -3, 'prox2_L5_pyramidal_nmda': -0.7,
        'prox1_mu': 16.5, 'prox1_sigma': 3, 'prox1_numspikes': 1.,
        'dist1_mu': 54, 'dist1_sigma': 4, 'dist1_numspikes': 2,
        'prox2_mu': 112, 'prox2_sigma': 18.5, 'prox2_numspikes': 1})

    constraints = dict({
        'prox1_L2_inhibitory_ampa': (-15, -1.8), 'prox1_L2_pyramidal_ampa': (-1.85, 0),
        'prox1_L5_inhibitory_ampa': (-15, -1.75), 'prox1_L5_pyramidal_ampa': (-2, 0),
        'prox1_L2_inhibitory_nmda': (-15, -2.), 'prox1_L2_pyramidal_nmda': (-1.6, 0),
        'prox1_L5_inhibitory_nmda': (-15, -1.75), 'prox1_L5_pyramidal_nmda': (-2, 0),
        'dist1_L2_inhibitory_ampa': (-15, 0), 'dist1_L2_pyramidal_ampa': (-6, 0),
        'dist1_L5_pyramidal_ampa': (-6, 0),
        'dist1_L2_inhibitory_nmda': (-15, 0), 'dist1_L2_pyramidal_nmda': (-6, 0),
        'dist1_L5_pyramidal_nmda': (-6, 0),
        'prox2_L2_inhibitory_ampa': (-15, 0), 'prox2_L2_pyramidal_ampa': (-15, -.5),
        'prox2_L5_inhibitory_ampa': (-15, 0), 'prox2_L5_pyramidal_ampa': (-15, -0.5),
        'prox2_L2_inhibitory_nmda': (-15, 0), 'prox2_L2_pyramidal_nmda': (-15, -.5),
        'prox2_L5_inhibitory_nmda': (-15, 0), 'prox2_L5_pyramidal_nmda': (-15, -0.5),
        'prox1_mu': (15, 25), 'prox1_sigma': (1, 4), 'prox1_numspikes': (0, 3),
        'dist1_mu': (50, 100), 'dist1_sigma': (1, 6), 'dist1_numspikes': (0, 3),
        'prox2_mu': (80, 150), 'prox2_sigma': (1, 20), 'prox2_numspikes': (0, 3)})

    return constraints, initial_params


# ---------------------------------------------------------------------------
# Windowed objective (rmse * (2 - corr) restricted to [win_tstart, win_tstop])
# ---------------------------------------------------------------------------
def _rmse_corr_evoked_windowed(initial_net, initial_params, set_params,
                               predicted_params, update_params, obj_values,
                               tstop, obj_fun_kwargs, best=None):
    """Same as _rmse_corr_evoked but the RMSE+corr comparison is restricted to
    [win_tstart, win_tstop] (read from obj_fun_kwargs), and non-finite losses
    (e.g. flat dipole -> pearsonr = nan) are replaced by a large finite penalty
    so a single degenerate candidate cannot poison the generation."""

    win_tstart = obj_fun_kwargs.get('win_tstart', 0.0)
    win_tstop = obj_fun_kwargs.get('win_tstop', tstop)
    nan_penalty = obj_fun_kwargs.get('nan_penalty', 1e6)

    def _safe_loss(dpl):
        val = _rmse_corr(dpl, obj_fun_kwargs['target'],
                         tstart=win_tstart, tstop=win_tstop,
                         weights=obj_fun_kwargs.get('weights', None))
        return val if np.isfinite(val) else nan_penalty

    is_batch = len(np.array(predicted_params).shape) == 2

    if is_batch:
        # CMA path: predicted_params is (popsize, n_params)
        predicted_params = np.array(predicted_params).reshape(-1, len(initial_params))
        params_batch = {name: predicted_params[:, idx]
                        for idx, name in enumerate(initial_params.keys())}

        new_net = initial_net.copy()

        def set_params_batch(a, b):
            set_params(b, a)

        batch_simulation = BatchSimulate(
            net=new_net, set_params=set_params_batch, save_outputs=False,
            save_dpl=True, dt=obj_fun_kwargs.get('dt', 0.025),
            n_trials=obj_fun_kwargs.get('n_trials', 1), tstop=tstop,
            overwrite=False, clear_cache=False,
            bsl_cor=obj_fun_kwargs.get('bsl_cor', None))

        res = batch_simulation.run(
            params_batch, n_jobs=obj_fun_kwargs.get('n_jobs', 1),
            combinations=False, backend='loky',
            verbose=obj_fun_kwargs.get('verbose', True))

        dpls = list()
        for batch_res in res['simulated_data']:
            for data in batch_res:
                # FIX: smooth & scale the actual simulated dipole (not the
                # accumulator list, which was the bug in _rmse_corr_evoked)
                _preprocess_dipole(data['dpl'], obj_fun_kwargs)
                dpls.append(average_dipoles(data['dpl']))

        obj = [_safe_loss(dpl) for dpl in dpls]

    else:
        # cobyla / bayesian path (single candidate)
        params = update_params(initial_params, predicted_params)
        new_net = initial_net.copy()
        set_params(new_net, params)

        dpls = simulate_dipole(new_net, tstop=tstop,
                               dt=obj_fun_kwargs.get('dt', 0.025),
                               n_trials=obj_fun_kwargs.get('n_trials', 1),
                               bsl_cor=obj_fun_kwargs.get('bsl_cor', None))
        _preprocess_dipole(dpls, obj_fun_kwargs)
        dpl = average_dipoles(dpls)
        obj = _safe_loss(dpl)

        if best is not None and obj < best['obj']:
            best['obj'] = obj
            best['params'] = predicted_params.copy()

    print(f"[{win_tstart:.0f}-{win_tstop:.0f} ms] "
          f"Mean Loss: {np.nanmean(obj):.3f}; Min Loss: {np.nanmin(obj):.3f}",
          flush=True)
    obj_values.append(obj)
    return obj


# ---------------------------------------------------------------------------
# Helpers for the iterative scheme
# ---------------------------------------------------------------------------
def _clip_to_bounds(params, constraints):
    """Clip parameter values into their constraint bounds (CMA's xbest can land
    marginally outside bounds, which would fail the Optimizer's range check)."""
    return {k: float(np.clip(v, constraints[k][0], constraints[k][1]))
            for k, v in params.items()}


def _make_set_params(fixed_params):
    """Return a set_params that injects fixed_params for the non-optimized drives
    and merges in the optimizer-supplied values for the current drive."""
    def _set(net, param_values):
        merged = dict(fixed_params)
        merged.update(param_values)
        set_params_opt_drives(net, merged)
    return _set


# ---------------------------------------------------------------------------
# Main iterative routine
# ---------------------------------------------------------------------------
def opt_cma_iterative(max_iter, popsize, n_procs, scaling_factor, smooth_win,
                      sigma0, n_trials=3, final_refine=False):

    data_url = f'{pathgen}MEG_detection_data/S1_SupraT.txt'
    exp_dpl = read_dipole(data_url)
    tstop = float(exp_dpl.times[-1])

    net_base = duecker_ET_model()
    set_up_drives_empty(net_base)
    dt = 0.05

    constraints, initial_params = get_constraints_init_params()

    # time window assigned to each drive (ms). prox2 runs to end of simulation.
    drive_windows = OrderedDict([
        ('prox1', (0.0, 45.0)),
        ('dist1', (45.0, 90.0)),
        ('prox2', (90.0, tstop)),
    ])

    # running best estimate of ALL parameters, refined drive by drive
    current_best = dict(initial_params)

    for drive_name, (win_t0, win_t1) in drive_windows.items():
        print(f"\n=== Optimizing {drive_name} over "
              f"[{win_t0:.0f}-{win_t1:.0f}] ms ===", flush=True)

        # parameters belonging to this drive only (note 'prox1_' vs 'prox2_'
        # so the prefixes do not collide)
        sub_keys = [k for k in constraints if k.startswith(drive_name + '_')]
        sub_constraints = {k: constraints[k] for k in sub_keys}
        sub_initial = _clip_to_bounds({k: current_best[k] for k in sub_keys},
                                      sub_constraints)

        # all other drives are held fixed at the current best
        fixed_params = {k: current_best[k] for k in current_best
                        if k not in sub_keys}
        set_params_chunk = _make_set_params(fixed_params)

        optim = Optimizer(net_base, tstop=tstop, constraints=sub_constraints,
                          solver='cma', set_params=set_params_chunk,
                          initial_params=sub_initial, max_iter=max_iter,
                          obj_fun=_rmse_corr_evoked_windowed)

        optim.fit(target=exp_dpl, n_trials=n_trials, scale_factor=scaling_factor,
                  sigma0=sigma0, smooth_window_len=smooth_win, dt=dt,
                  popsize=popsize, pth_backup=pth_backup, bsl_cor='duecker',
                  n_jobs=n_procs, verbose=False,
                  win_tstart=win_t0, win_tstop=win_t1)

        # write this drive's optimized values back into the running best
        for key, val in zip(sub_initial.keys(), optim.opt_params_):
            current_best[key] = float(val)

        with open(f'opt_ERP_cma_{popsize}_{drive_name}.pkl', 'wb') as f:
            pickle.dump({'optim': optim, 'current_best': dict(current_best),
                         'window': (win_t0, win_t1)}, f)
        print(f"Finished {drive_name}; running best updated.", flush=True)

    # optional final joint refinement over the full window (all drives free),
    # mirroring the "last chunk" cleanup in the legacy optimize_evoked
    if final_refine:
        print("\n=== Final joint refinement over full window ===", flush=True)
        sub_initial = _clip_to_bounds(current_best, constraints)
        set_params_chunk = _make_set_params({})  # nothing fixed

        optim = Optimizer(net_base, tstop=tstop, constraints=constraints,
                          solver='cma', set_params=set_params_chunk,
                          initial_params=sub_initial, max_iter=max_iter,
                          obj_fun=_rmse_corr_evoked_windowed)

        optim.fit(target=exp_dpl, n_trials=n_trials, scale_factor=scaling_factor,
                  sigma0=sigma0, smooth_window_len=smooth_win, dt=dt,
                  popsize=popsize, pth_backup=pth_backup, bsl_cor='duecker',
                  n_jobs=n_procs, verbose=False,
                  win_tstart=0.0, win_tstop=tstop)

        for key, val in zip(sub_initial.keys(), optim.opt_params_):
            current_best[key] = float(val)
        with open(f'opt_ERP_cma_{popsize}_final.pkl', 'wb') as f:
            pickle.dump({'optim': optim, 'current_best': dict(current_best)}, f)

    with open(f'opt_ERP_cma_{popsize}_iterative_best.pkl', 'wb') as f:
        pickle.dump(current_best, f)
    print("\nIterative optimization complete.", flush=True)
    return current_best


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Iterative (drive-by-drive) CMA optimization pipeline.')
    parser.add_argument('--max_iter', type=int, default=1,
                        help='Max CMA iterations per drive')
    parser.add_argument('--popsize', type=int, default=16,
                        help='CMA population size')
    parser.add_argument('--n_procs', type=int, default=20,
                        help='Number of parallel jobs')
    parser.add_argument('--scale_factor', type=int, default=1000,
                        help='Dipole scale factor')
    parser.add_argument('--smooth_window_len', type=int, default=5,
                        help='Smoothing window length')
    parser.add_argument('--sigma0', type=float, default=0.05,
                        help='Initial CMA sigma0 (scaled by bound width)')
    parser.add_argument('--n_trials', type=int, default=3,
                        help='Trials averaged per candidate')
    parser.add_argument('--final_refine', action='store_true', default=False,
                        help='Run a final joint full-window refinement step')
    args = parser.parse_args()

    opt_cma_iterative(max_iter=args.max_iter,
                      popsize=args.popsize,
                      n_procs=args.n_procs,
                      scaling_factor=args.scale_factor,
                      smooth_win=args.smooth_window_len,
                      sigma0=args.sigma0,
                      n_trials=args.n_trials,
                      final_refine=args.final_refine)
