"""Bursty gamma simulation for the duecker_ET_model (manuscript Fig. 11h–k).

Runs the bursty-gamma network over many trials, and pickles the network and dipoles to ``sim_results/``. 
Because this is a multi-trial (default 100) simulation it is meant to be run on the cluster
(submitted to Slurm via ``step1_gamma_burst_simulation.sh``); ``step2_inspect_gamma_multiscale.ipynb`` then
loads the pickles and plots them with ``plot_multi_trial_sim``.

The network and its drives come from the shared network configuration
``duecker_ET_gamma_50Hz_bursty`` (built by ``network_sim/make_network_configs.py``).

Usage
-----
    python step1_gamma_burst_simulation.py --n_trials 100 --n_procs 50

Author: Katharina Duecker <katharina.duecker@gmail.com>
"""

import argparse
import os
import os.path as op
import pickle

import hnn_core
from hnn_core import JoblibBackend, read_network_configuration, simulate_dipole

from hnn_tuning import NETWORK_SIM, RESULTS

NETWORK_CONFIG = 'duecker_ET_gamma_50Hz_bursty'

out_pth = op.join(RESULTS, "gamma_sim")
os.makedirs(out_pth, exist_ok=True)


def run_main(n_trials, n_procs, tstop=1000.):
    """Simulate the bursty-gamma network and pickle the network + dipoles."""

    CONFIG_ROOT = op.join(NETWORK_SIM, 'network_configs')
    net = read_network_configuration(op.join(CONFIG_ROOT, NETWORK_CONFIG))

    with JoblibBackend(n_jobs=n_procs):
        dpl = simulate_dipole(net, tstop=tstop, bsl_cor='duecker', n_trials=n_trials)

    netfilename = op.join(out_pth, 'net_gamma_bursty.pkl')
    dplfilename = op.join(out_pth, 'dpl_gamma_bursty.pkl')

    with open(netfilename, 'wb') as file:
        pickle.dump(net, file)
    with open(dplfilename, 'wb') as file:
        pickle.dump(dpl, file)

    print(f'wrote {netfilename}\nwrote {dplfilename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bursty gamma simulation for the duecker_ET_model (Fig. 11h-k).')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials to simulate (default: 100)')
    parser.add_argument('--n_procs', type=int, default=50,
                        help='Number of parallel jobs for the Joblib backend (default: 50)')
    parser.add_argument('--tstop', type=float, default=1000.,
                        help='Simulation duration in ms (default: 1000)')
    args = parser.parse_args()

    run_main(args.n_trials, args.n_procs, tstop=args.tstop)
