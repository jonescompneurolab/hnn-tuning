"""Build the shared network-configuration files from the legacy param JSONs.

MAINTENANCE / PROVENANCE SCRIPT — you do not need to run this to reproduce any
figure. The simulations load ready-made network configurations via
``read_network_configuration``; this script is the record of how those
configuration files were produced.

Background: the models used to be specified as a params JSON (read with
``hnn_core.read_params``) plus drive definitions written inline in each
notebook/script. Network configuration files bundle both (network + drives) into
a single shareable file, so that is what the compendium now ships. Running this
script regenerates every configuration from its params JSON + the drive
parameters recorded below, and writes them into the hnn-core fork's
``network_configs/`` directory.

Usage
-----
    python make_network_configs.py --list          # show what would be built
    python make_network_configs.py --all           # build everything
    python make_network_configs.py --name duecker_ET_gamma_50Hz_sustained

Author: Katharina Duecker <katharina.duecker@gmail.com>
Part of the hnn-tuning research compendium.
"""

import argparse
import os
import os.path as op

import hnn_core
from hnn_core import duecker_ET_model, write_network_configuration

HNN_CORE_ROOT = op.dirname(hnn_core.__file__)
PARAM_ROOT = op.join(HNN_CORE_ROOT, 'param')
CONFIG_ROOT = op.join(HNN_CORE_ROOT, 'network_configs')


# --- drive builders -------------------------------------------------------
# One function per model. Each takes a Network and attaches the drives that
# belong to that configuration, using the values from the original notebooks.

def _drives_gamma_sustained(net):
    """Sustained ~50 Hz gamma: strong, fast Poisson drive (gamma.ipynb)."""
    net.add_poisson_drive(
        'prox_poiss', tstart=0, tstop=1000,
        rate_constant={'L2_pyramidal': 200.0, 'L5_pyramidal': 200.0},
        weights_ampa={'L2_pyramidal': 0.0045, 'L5_pyramidal': 0.0055},
        weights_nmda={'L2_pyramidal': 0.0045, 'L5_pyramidal': 0.0055},
        location='proximal',
        synaptic_delays={'L2_pyramidal': 0.1, 'L5_pyramidal': 1.0},
        event_seed=1349)


def _drives_gamma_bursty(net):
    """Bursty gamma: reduced inhibition, lower drive rate and NMDA (gamma.ipynb)."""
    net.add_poisson_drive(
        'prox_poiss', tstart=0, tstop=1000,
        rate_constant={'L2_pyramidal': 100.0, 'L5_pyramidal': 100.0},
        weights_ampa={'L2_pyramidal': 0.0035, 'L5_pyramidal': 0.0055},
        weights_nmda={'L2_pyramidal': 0.0025, 'L5_pyramidal': 0.0035},
        location='proximal',
        synaptic_delays={'L2_pyramidal': 0.1, 'L5_pyramidal': 1.0},
        event_seed=1349)


def _drives_alpha_beta(net):
    """Alpha/beta complex: proximal feedforward + delayed distal feedback bursty
    drives (step1_alpha_beta_complex_simulation.sim_analysis_params_default).

    NOTE: the per-run seeds are set from the Slurm array index at run time
    (``prox_seed = run_idx - 1 + 544``, ``dist_seed = run_idx - 1 + 476``); the
    defaults are baked in here.
    """
    fb_delay = 5
    net.add_bursty_drive(
        'prox', tstart=50., burst_rate=10, burst_std=20, numspikes=2,
        spike_isi=10, n_drive_cells=10, location='proximal',
        weights_ampa={'L2_inhibitory': 0.0, 'L2_pyramidal': 2.5e-4,
                      'L5_pyramidal': 2.5e-4, 'L5_inhibitory': 0.0},
        synaptic_delays={'L2_inhibitory': 0.1, 'L2_pyramidal': 0.1,
                         'L5_pyramidal': 0.1, 'L5_inhibitory': 0.1},
        event_seed=180)
    net.add_bursty_drive(
        'dist', tstart=50. + fb_delay, burst_rate=10, burst_std=15, numspikes=2,
        spike_isi=10, n_drive_cells=10, location='distal',
        weights_ampa={'L2_inhibitory': 0, 'L5_pyramidal': 2.5e-4 * 4,
                      'L2_pyramidal': 2.5e-4 * 4},
        synaptic_delays={'L2_inhibitory': 0.1, 'L5_pyramidal': 0.1,
                         'L2_pyramidal': 0.1},
        event_seed=120)


# --- registry -------------------------------------------------------------
# name -> (params JSON in hnn_core/param/, drive builder or None)
# `None` params  = build from the default duecker_ET_model()
# `None` drives  = network only, no drives attached
CONFIGS = {
    'duecker_ET_gamma_50Hz_sustained': ('duecker_ET_gamma_50Hz_sustained.json',
                                        _drives_gamma_sustained),
    'duecker_ET_gamma_50Hz_bursty': ('duecker_ET_gamma_50Hz_bursty.json',
                                     _drives_gamma_bursty),
    'duecker_ET_alpha_beta_complex': (None, _drives_alpha_beta),
    'duecker_ET_default': (None, None),
}

# NOT built here: the supra-threshold ERP configuration
# (`duecker_ET_supra_ERP`). Its drives come out of the CMA-ES optimisation, so it
# is written directly from the fitted network in
# network_sim/ERP/step3_inspect_ERP_multiscale.ipynb, not rebuilt from a params file.


def build(name, overwrite=False):
    """Build one network configuration and write it to hnn-core's network_configs/."""
    params_fname, add_drives = CONFIGS[name]

    if params_fname is None:
        net = duecker_ET_model()
    else:
        params_path = op.join(PARAM_ROOT, params_fname)
        if not op.exists(params_path):
            raise FileNotFoundError(f'params JSON not found: {params_path}')
        net = duecker_ET_model(hnn_core.read_params(params_path))

    if add_drives is not None:
        add_drives(net)

    os.makedirs(CONFIG_ROOT, exist_ok=True)
    out_path = op.join(CONFIG_ROOT, name)
    if op.exists(out_path) and not overwrite:
        print(f'  SKIP {name} (exists; pass --overwrite to replace)')
        return out_path

    write_network_configuration(net, out_path)
    print(f'  wrote {out_path}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rebuild the shared network configuration files from the legacy param JSONs.')
    parser.add_argument('--name', type=str, default=None,
                        help='Build a single configuration by name')
    parser.add_argument('--all', action='store_true',
                        help='Build every configuration in the registry')
    parser.add_argument('--list', action='store_true',
                        help='List the configurations and their source params files')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite configuration files that already exist')
    args = parser.parse_args()

    if args.list:
        print(f'network_configs/ -> {CONFIG_ROOT}\n')
        for name, (params_fname, add_drives) in CONFIGS.items():
            src = params_fname or 'duecker_ET_model() default'
            drives = add_drives.__name__ if add_drives else 'no drives'
            print(f'  {name}\n      params: {src}\n      drives: {drives}')
    elif args.all:
        for name in CONFIGS:
            build(name, overwrite=args.overwrite)
    elif args.name:
        if args.name not in CONFIGS:
            raise SystemExit(f'unknown config {args.name!r}; choose from {list(CONFIGS)}')
        build(args.name, overwrite=args.overwrite)
    else:
        parser.print_help()
