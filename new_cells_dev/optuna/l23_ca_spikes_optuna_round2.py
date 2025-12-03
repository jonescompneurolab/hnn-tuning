"""Narrow down parameters that provide good fit to experimentally observed waveform shapes"""

import os

main_pth = '/users/kduecker/hnn-tuning'
pth_mod = os.path.join(main_pth, 'local_hnn/hnn_core/mod')
pth_rslts = os.path.join(main_pth, 'new_cells_dev/optuna/ca_spikes')
os.makedirs(pth_rslts, exist_ok=True)
pth_data = os.path.join(main_pth, 'data/l23')

from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

import numpy as np
from functools import partial
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.signal import windows
from scipy.optimize import curve_fit

import optuna
from multiprocessing import Pool, Lock
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial
import threading
import sqlite3
import json

import pickle

import multiprocessing
import argparse

import numpy as np
import pickle
import os
from functools import partial
import random

# own utils
# Load custom modules

import sys
sys.path.append(os.path.join(main_pth, 'new_cells_dev/utils'))
from cell_builder_utils import l2_params, _get_dends, _get_pyr_soma, Cell
from iascone_mechs_utils import Mechs, get_iascone_params


def get_sections_load_mechs(params):

    section_params, end_pts, cell_tree = l2_params()

    v_init = {'all': -65}

    # replace cm and Ra
    for p_key, p_val in params['passive_params'].items():
        for key in section_params:
            if p_key in key:
                section_params[key] = p_val

    # build sections
    section_names = list(end_pts.keys())

    sections = _get_dends(section_params, 'L2Pyr', section_names, end_pts, v_init=v_init)
    sections['soma'] = _get_pyr_soma(section_params, 'L2Pyr', end_pts['soma'], v_init=v_init['all'])

    mech_class = Mechs(pth_mod)

    return sections, mech_class, cell_tree


def flatten_params(nested_params):
    """Flatten nested dictionary into a list of values."""
    flattened = []
    # Flatten compartment conductances
    for comp in ['soma', 'basal', 'dend']:
        for channel, value in nested_params[comp].items():
            if isinstance(value, dict):  # Handle nested CaDynamics_E2
                for param, sub_value in value.items():
                    flattened.append(sub_value)
            else:
                flattened.append(value)
    # Flatten top-level keys
    flattened.append(nested_params['e_pas'])
    # Flatten passive_params
    for param in ['cm', 'Ra']:
        flattened.append(nested_params['passive_params'][param])
    return flattened

def reconstruct_params(flat_list):
    """Reconstruct nested dictionary from flattened list."""
    # Template structure with original keys (values will be replaced)
    reconstructed = {
        'soma': {'NaTs2_t_32d': None, 'SKv3_1': None, 'Nap_Et2': None, 
                 'Ca_HVA': None, 'Ca_LVAst': None, 'pas': None, 'Ih': None, 'SK_E2': None, 'Im': None},
        'basal': {'NaTs2_t_32d': None, 'SKv3_1': None, 'Ih': None, 'pas': None},
        'dend': {'NaTa_t_32d': None, 'SKv3_1': None, 'Ca_HVA': None, 'Ca_LVAst': None,
                 'pas': None, 'SK_E2': None, 'Ih': None, 'Im': None,
                 'CaDynamics_E2': {'decay': None, 'gamma': None}},
        'e_pas': None,
        'passive_params': {'L2Pyr_soma_cm': None, 'L2Pyr_soma_Ra': None,
                           'L2Pyr_dend_cm': None, 'L2Pyr_dend_Ra': None},
        'Ca_HVA_fun': {'gdend': None, 'xkink': None, 'hot_low': None, 'hot_up': None, 'hot_fact': None},
        'Ca_LVA_fun': {'gdend': None, 'xkink': None, 'hot_low': None, 'hot_up': None, 'hot_fact': None}
    }
    
    # Index tracker for flat_list
    index = 0
    ca_index = 0            # index for ca function
    # Reconstruct compartments
    for comp in ['soma', 'basal', 'dend']:
        for channel in reconstructed[comp]:
            if channel == 'CaDynamics_E2':
                    for sub_param in reconstructed[comp][channel]:
                        reconstructed[comp][channel][sub_param] = flat_list[index]
                        index += 1
            if (comp == 'dend' and channel == 'Ca_HVA' and len(ca_fun_vals)>0) or (comp == 'dend' and channel == 'Ca_LVAst' and len(ca_fun_vals)>0):
                reconstructed[comp][channel] = partial(linear_gbar,flat_list[index], ca_fun_vals[ca_index], ca_fun_vals[ca_index+1], hotzone=[ca_fun_vals[ca_index+2], ca_fun_vals[ca_index+3]], hotzone_factor=ca_fun_vals[ca_index+4])
                index += 1
                ca_index +=5
            else:
                reconstructed[comp][channel] = flat_list[index]
                index += 1
    # Reconstruct e_pas and passive_params
    reconstructed['e_pas'] = flat_list[index]
    index += 1
    for param in reconstructed['passive_params']:
        if 'cm' in param:
            reconstructed['passive_params'][param] = flat_list[-2]
        elif 'Ra' in param:
            reconstructed['passive_params'][param] = flat_list[-1]
        
    return reconstructed

def ca_spike(params, ca_vals=[]):

    # neuron imports
    import neuron
    from neuron import h
    from neuron.units import ms, mV

    neuron.load_mechanisms(pth_mod)
    h.load_file("stdrun.hoc")

    # Get morphological and conductance parameters
    sections, mech_class, cell_tree = get_sections_load_mechs(params)
    
    # Build cell 
    l2pyr= Cell(sections, cell_tree, cell_type='L2_pyramdial', gid=0)
    l2pyr.build()
    l2pyr = mech_class.add_mechs(l2pyr, params)

    stim_sect = 'apical_1'
    stim = h.IClamp(l2pyr._nrn_sections[stim_sect](0.5))
    stim.delay = 20
    stim.dur = 1000
    stim.amp = 0.25

    t = h.Vector().record(h._ref_t)

    stim_v = h.Vector().record(l2pyr._nrn_sections[stim_sect](0.5)._ref_v)

    h.celsius = 37
    h.finitialize()
    h.continuerun(1100 * ms)

    t_np = t.as_numpy()
    v_np = stim_v.as_numpy()

    pk = find_peaks(v_np, height=-10, width=0.1, rel_height=0.5)

    return pk

def loss_function(proposed_vals, ca_vals=[]):

    # bring params in the right format
    params_dict = reconstruct_params(proposed_vals, ca_vals)

    dend_peak = ca_spike(params_dict)

    loss = (len(dend_peak[0])-4)**2

    return loss

def objective(trial):

    print(f"Running trial {trial.number=}")

    param_space = [[1.5, 6],
                [0.4, 2],
                [5e-5, 5e-3],
                [1e-5, 5e-3],
                [1e-5, 5e-3],
                [1/40000, 1/9000],
                [1e-5, 5e-3], 
                [1e-5, 5e-3],
                [5e-4, 5e-2],
                [1e-4, 1e-2],
                [1e-4, 1e-2],
                [1e-4, 1e-2],
                [1/40000, 1/9000],
                [1e-5, 5e-3],
                [1e-5, 5e-2],
                [1e-5, 5e-3],
                [1e-5, 5e-3],
                [1/40000, 1/9000], 
                [1e-7, 5e-4],
                [1e-5, 5e-3],
                [1e-3, 5e-2],
                [60, 300], 
                [1e-4, .1],
                [-90, -65],
                [0.5, 3],
                [70, 200]]
                
    param_names = ['NaTs2_t_32d_soma','SKv3_1_soma', 'Nap_Et2_soma', 'Ca_HVA_soma', 'Ca_LVAst_soma', 'pas_soma', 'Ih_soma', 'SK_E2_soma', 'Im_soma', 'NaTs2_t_32d_basal', 'SKv3_1_basal', 'Ih_basal', 'pas_basal',
                        'NaTa_t_32d_apical', 'SKv3_1_apical','Ca_HVA_apical', 'Ca_LVAst_apical', 'pas_apical', 'SK_E2_apical', 'Ih_apical', 'Im_apical','decay', 'gamma',
                        'e_pas', 'cm', 'Ra']
    proposed_vals = []
    for key, val in zip(param_names, param_space):
        proposed_vals.append(trial.suggest_float(key, val[0], val[1]))  

    ca_param_names = ['ca_hva_dend', 'ca_hva_xkink', 'ca_hva_hot_low', 'ca_hva_hot_high', 'ca_hva_hot_fact', 'ca_lva_dend', 'ca_lva_xkink', 'ca_lva_hot_low', 'ca_lva_hot_high', 'ca_lva_hot_fact']
    ca_param_space = [[1e-5, 5e-3],
                        [200, 600],
                        [100, 300],
                        [100, 600],
                        [1, 10],
                        [1e-5, 5e-3],
                        [200, 600],
                        [100, 300],
                        [100, 600],
                        [1, 10]]
    proposed_ca = []
    # for key, val in zip(ca_param_names, ca_param_space):
    #     proposed_ca.append(trial.suggest_float(key, val[0], val[1]))  

    loss = loss_function(proposed_vals, proposed_ca)  
    return loss

def run_optimization(study_name, journal_file, sampler, process_id):
    # Each process gets its own study instance but shares the same storage
    study = optuna.create_study(
        study_name=study_name, 
        direction='minimize', 
        sampler=sampler,
        storage=JournalStorage(JournalFileBackend(
            file_path=journal_file
        )), 
        load_if_exists=True
    )
    
    # Run trials
    study.optimize(objective, n_trials=10000)
    
    return study.best_value, study.best_params

def main(processes, sampler_name):
    # Map sampler names to sampler classes
    sampler_map = {
        'random': optuna.samplers.RandomSampler(),
        'tpe': optuna.samplers.TPESampler(),
        'cmaes': optuna.samplers.CmaEsSampler(),
        # Add other samplers as needed
    }
    
    if sampler_name not in sampler_map:
        raise ValueError(f"Unknown sampler: {sampler_name}. Available options: {list(sampler_map.keys())}")
    
    sampler = sampler_map[sampler_name] 
    # Dynamically generate file paths based on the sampler name
    study_name = f'l23_ca_spike_{sampler_name.upper()}'
    journal_file = os.path.join(pth_rslts, f"journal_ca_spike_{sampler_name}.log")
    complete_study_file = os.path.join(pth_rslts, f'complete_study_ca_spike_{sampler_name}.pkl')
    study_df_file = os.path.join(pth_rslts, f'study_df_{sampler_name}.pkl')
    
    # Create a partial function with fixed arguments
    optimization_func = partial(run_optimization, study_name, journal_file, sampler)
    
    # Run optimization with the specified number of processes
    with Pool(processes=processes) as pool:
        pool.map(optimization_func, range(processes))

    study = optuna.create_study(
    study_name=study_name,
    direction='minimize',
    storage=JournalStorage(JournalFileBackend(file_path=journal_file)),
    load_if_exists=True)

    # Save the study and DataFrame
    with open(complete_study_file, 'wb') as f:
        pickle.dump(study, f)

    df = study.trials_dataframe()
    df.to_pickle(study_df_file)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run Optuna optimization with configurable processes and sampler.')
    parser.add_argument('--processes', type=int, default=30, help='Number of processes to use (default: 30)')
    parser.add_argument('--sampler_name', type=str, default='cmaes', help='Sampler to use (e.g., random, tpe, cmaes; default: cmaes)')
    
    args = parser.parse_args()
    
    os.makedirs(pth_rslts, exist_ok=True)
    
    main(args.processes, args.sampler_name.lower())