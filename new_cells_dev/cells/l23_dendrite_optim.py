
"""Dendrite optimization"""

# set up
import optuna
from multiprocessing import Pool, Lock
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial
import threading
import sqlite3
import json
import pandas
import argparse

import neuron
from neuron import h
import matplotlib.pyplot as plt
from neuron.units import ms, mV
import numpy as np
import pickle
import os
from functools import partial

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import signal
# Load custom modules
main_pth = '/users/kduecker/hnn-tuning'
pth_mod = os.path.join(main_pth, 'local_hnn/hnn_core/mod')
pth_rslts = os.path.join(main_pth, 'new_cells_dev/cells')
import sys
sys.path.append(os.path.join(main_pth, 'new_cells_dev/utils'))
from cell_builder_utils import l2_params, _get_dends, _get_pyr_soma, Cell
from iascone_mechs_utils import Mechs, get_iascone_params, Ih_condu_apical, distribute_gbar
from hay_mechs_utils import linear_gbar



# target attenuation from literature

soma_dend_atten = {
    'bAp': 290, # Gidon, 2020 Fig. S5
    'bAp_upstroke': 121,
    'sEPSP': 613, # Gooch, 2022
    'hyperpol': np.mean((475, 620)), #Gidon, 2020 Fig. S1; Gooch, 2022
}


def exp_decay(x, lambda_val, A):
    return A * np.exp(-x / lambda_val)

def get_sections_load_mechs():

    section_params, end_pts, cell_tree = l2_params()

    v_init = {'soma': -74.36601977929463,
            'basal_1': -74.3646051889625,
            'basal_2': -74.36055185255331,
            'basal_3': -74.36055185255331,
            'apical_oblique': -74.3569656391541,
            'apical_trunk': -74.36313736979325,
            'apical_1': -74.35099846351194,
            'apical_tuft': -74.34082715755618}

    # build sections
    section_names = list(end_pts.keys())

    sections = _get_dends(section_params, 'L2Pyr', section_names, end_pts, v_init=v_init)
    sections['soma'] = _get_pyr_soma(section_params, 'L2Pyr', end_pts['soma'], v_init=v_init['soma'])

    mech_class = Mechs(pth_mod)

    return sections, mech_class, cell_tree

def attenuation_from_section(ref_sect, params, amp, stim_dur, stim_l):

    section_names = ['soma', 'apical_trunk', 'apical_1', 'apical_tuft']
    amp_ref = []
    amp_dend = []

    distance = []
    i = 0

    sections, mech_class, cell_tree = get_sections_load_mechs()
    l2pyr= Cell(sections, cell_tree, cell_type='L2_pyramidal', gid=0)
    l2pyr.build()
    l2pyr = mech_class.add_mechs(l2pyr, params)
    h.distance(sec = l2pyr._nrn_sections[ref_sect])
    
    
    stim = h.IClamp(l2pyr._nrn_sections[ref_sect](0.5))
    stim.delay = 20
    stim.dur = stim_dur
    stim.amp = amp

    t = h.Vector().record(h._ref_t)
    
    ref_voltg = h.Vector().record(l2pyr._nrn_sections[ref_sect](0.5)._ref_v)
    
    sections_minus_ref = [section for section in section_names if section != ref_sect]

    upstroke = []
    for s, section in enumerate(sections_minus_ref):
        for sg, seg in enumerate(l2pyr._nrn_sections[section]):
            
            dend_voltg = h.Vector().record(l2pyr._nrn_sections[section](seg.x)._ref_v)
        
            distance.append(h.distance(seg))
            
            h.celsius = 37
            h.finitialize()
            h.continuerun(stim_l * ms)
            
            # low-pass filter 5KHz like in Gidon et al.,
            sfreq = 1000/np.diff(t.as_numpy())[0]
            b, a = signal.butter(0, 5000, btype='low', analog=False, fs=sfreq)

            v = dend_voltg.as_numpy()
            d_filt = signal.lfilter(b, a, v)
            r_filt = signal.lfilter(b, a, ref_voltg.as_numpy())
            t_np = t.as_numpy()

            if amp > 0:  
                # upstroke per ms (Gidon et al., 2020 Fig S5 A)
                upstroke.append(np.max(np.diff(d_filt))/np.diff(t_np)[0])     

                amp_ref.append(np.max(r_filt) - np.min(r_filt))
                amp_dend.append(np.max(d_filt) - np.min(d_filt))

            elif amp     < 0:
                start_idx = np.argmin(np.abs(t_np - (stim.delay+800))) # Gidon 2020
                end_idx = np.argmin(np.abs(t_np - (stim.delay+1000))) # Gooch, 2022

                amp_ref.append(np.mean(r_filt[start_idx:end_idx]))
                amp_dend.append(np.mean(d_filt[start_idx:end_idx]))

            i += 1

    # calculate attenuation
    atten = np.array(amp_dend) / np.array(amp_ref) * 100

    return atten, distance, upstroke

def objective(trial):


    print(f"Running trial {trial.number=} in process {os.getpid()}")

    ca_hva_gbar = trial.suggest_float('ca_hva_gbar', 0, 2e-3)

    ca_lva_gbar = trial.suggest_float('ca_lva_gbar', 0, 2e-3)

    Rm = trial.suggest_int('Rm', 5000, 40000)
    
    suggest_params = {'soma': {'NaTs2_t_32d': 20_400e-4,
                                'SKv3_1': 6_930e-4,
                                'Nap_Et2': 1e-4/2,
                                'Ca_HVA': 9.92e-04,
                                'Ca_LVAst': 34.3e-4*0.1,
                                
                                #'SK_E2': 0.099433/2*0.001,
                                'pas': 1.0 / 10000.0,
                                'Ih': 0.000080*2,
                                'Im': 8e-3*0.01},
                        'basal': {'NaTs2_t_32d': 0.008009*0.01,#partial(linear_gbar,0.008009/2,0,255),
                                  'SKv3_1': 0.000513*0.01,#partial(linear_gbar,0,6_930e-4*0.1,255),
                                  'Ih': 0.00008,
                                  'pas':  1.0 / 10000.0},
                            'dend': {'NaTa_t_32d': trial.suggest_float('NaTa_t_32d_apical', 0, 1e-2),
                                    'SKv3_1': trial.suggest_float('SKv3_1_apical', 0, 1e-2),
                                    'Ca_HVA': trial.suggest_float('ca_hva_gbar', 0, 2e-3), 
                                    'Ca_LVAst': trial.suggest_float('ca_lva_gbar', 0, 2e-3), 
                                    'pas': 1/Rm,
                                    'SK_E2': trial.suggest_float('SK_E2_apical', 0, 1e-4),
                                    'Ih': trial.suggest_float('Ih_gbar', 0, 1e-5),
                                    'Im': trial.suggest_float('Im_apical', 0, 1e-2),
                                    'CaDynamics_E2': {'decay': 342.544232, 'gamma': 0.000533*2.7}},
                            'e_pas': -75}


    spike_amp, elec_sep, upstroke = attenuation_from_section('soma', suggest_params, amp=2, stim_dur=5, stim_l=100)
    popt, pcov = curve_fit(exp_decay, np.array(elec_sep), np.array(spike_amp), p0=[100, spike_amp[0] ])
    lambda_spike_amp = popt[0]
    popt, pcov = curve_fit(exp_decay, np.array(elec_sep), np.array(upstroke), p0=[100, upstroke[0] ])
    lambda_upstroke = popt[0]

    hyperpol_amp, elec_sep, _ = attenuation_from_section('soma', suggest_params, amp=-0.3, stim_dur=1000, stim_l=1020)
    popt, pcov = curve_fit(exp_decay, np.array(elec_sep), np.array(hyperpol_amp), p0=[100, hyperpol_amp[0]])
    lambda_hyperpol_amp = popt[0]

    loss = (np.abs(soma_dend_atten['bAp'] - lambda_spike_amp) + np.abs(soma_dend_atten['bAp_upstroke'] - lambda_upstroke) + np.abs(soma_dend_atten['hyperpol'] - lambda_hyperpol_amp))

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
    study.optimize(objective, n_trials=1000)
    
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
    study_name = f'l23_dendrite_{sampler_name.upper()}'
    journal_file = os.path.join(pth_rslts, f"journal_dendrite_{sampler_name}.log")
    complete_study_file = os.path.join(pth_rslts, f'complete_study_dendrite_{sampler_name}.pkl')
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

