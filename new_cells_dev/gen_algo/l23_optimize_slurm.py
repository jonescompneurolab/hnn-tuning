"""Optimize L2/3 cells
Soma: Allen Cell Map data
Dendrite: Gidon, 2020; Gooch, 2022""" 

import os

main_pth = '/users/kduecker/hnn-tuning'
pth_mod = os.path.join(main_pth, 'local_hnn/hnn_core/mod')
pth_rslts = os.path.join(main_pth, 'new_cells_dev/gen_algo/results_l23')
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

import pygad
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


import optuna
from multiprocessing import Pool, Lock
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial



def get_targets():

    soma_dend_atten = {
        'bAp': 290, # Gidon, 2020 Fig. S5
        'bAp_upstroke': 121,
        'sEPSP': 613, # Gooch, 2022
        'hyperpol': np.mean((475, 620)), #Gidon, 2020 Fig. S1; Gooch, 2022
        'inp_res': 40
    }

    with open(os.path.join(pth_data,'ephys_l2.pkl'), 'rb') as f:
        l2_allen = pickle.load(f)

    with open(os.path.join(pth_data,'ephys_l3.pkl'), 'rb') as f:
        l3_allen = pickle.load(f)

    return soma_dend_atten, l2_allen, l3_allen

def _exp_decay(x, lambda_val, A):
    return A * np.exp(-x / lambda_val)

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

def _get_start_params():
    """Hand-tuned + optuna optimized starting values for the conductances"""

    return {'soma': {'NaTs2_t_32d': 20_400e-4*2,
                                'SKv3_1': 6_930e-4*1.5,
                                'Nap_Et2': 1e-4/2*10,
                                'Ca_HVA': 9.92e-04,
                                'Ca_LVAst': 34.3e-4*0.1,
                                'pas': 1.0 / 38331,
                                'Ih': 0.000080*10,
                                'SK_E2': 5e-02,
                                'Im': 8e-3*0.001},
                        'basal': {'NaTs2_t_32d': 0.008009,
                                  'SKv3_1': 0.000513,
                                  'Ih': 0.00008*10,
                                  'pas':  1.0 / 30000.0},
                        'dend': {'NaTa_t_32d': 0.0007545577135222227,
                                'SKv3_1': 0.008911542253622935,
                                'Ca_HVA': 0.0010229989264147366,
                                'Ca_LVAst': 0.0010793073879112331,
                                'pas': 1.0 /29062,
                                'SK_E2': 5.183204947473679e-05,
                                'Ih': 0.000256109816290954,
                                'Im': 0.004888309098371485,
                                'CaDynamics_E2': {'decay': 132.59067310238868, 'gamma': 0.0025792496905644342}},
                        'e_pas': -75,
                        'passive_params': {'cm': 2,
                                           'Ra': 200}}

def _exp_func(t, tau, A, C):
    return A * (1 - np.exp(-t / tau)) + C

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
                           'L2Pyr_dend_cm': None, 'L2Pyr_dend_Ra': None}
    }
    
    # Index tracker for flat_list
    index = 0
    # Reconstruct compartments
    for comp in ['soma', 'basal', 'dend']:
        for channel in reconstructed[comp]:
            if channel == 'CaDynamics_E2':
                for sub_param in reconstructed[comp][channel]:
                    reconstructed[comp][channel][sub_param] = flat_list[index]
                    index += 1
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



# initialize neuron cell


"""In silico experiments"""

def rmp(params):
    """Resting membrane potential"""

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

    try:
        t = h.Vector().record(h._ref_t)
        soma_voltg = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)

        
        h.celsius = 37
        h.finitialize(-65*mV)
        h.continuerun(100 * ms)
        soma_v = soma_voltg.as_numpy()
        rmp = soma_v[-3]

        pk = find_peaks(soma_v, height=-40) # penalize if there is an AP without stim

        if np.isnan(rmp):
            rmp = 10000
        elif len(pk[0])>0:
            rmp = 20000

    except Exception as e:
        print(f"rmp ERROR: {str(e)}", flush=True)
        return 1000
    del l2pyr

    return rmp

def input_res(params):
    """Input resistance"""

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

    impedance = h.Impedance()
    h.celsius = 37
    h.finitialize()
    impedance.loc(0.5, sec=l2pyr._nrn_sections['soma'])
    impedance.compute(0)

    inp_res = impedance.input(l2pyr._nrn_sections['soma'](0.5))

    return inp_res

def ap_measures(params):
    """Action potential measures"""

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

    stim = h.IClamp(l2pyr._nrn_sections['soma'](0.5))
    stim.delay = 5
    stim.dur = 3
    stim.amp = 2
    clamp_current = h.Vector().record(stim._ref_i)

    t = h.Vector().record(h._ref_t)
    v_soma = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)

    h.celsius = 37
    h.dt = .01
    h.finitialize()
    h.continuerun(30 * ms)

    v_np = v_soma.as_numpy()
    t_np = t.as_numpy()
    ap, properties = find_peaks(v_np, height=0, width=0)

    if len(ap) == 0:
        # return these values when there is no AP
        ap_thresh = 0
        ap_width = 100
        ap_up_down = 0

        return ap_thresh, ap_width, ap_up_down

    elif len(ap) > 1: 
        #return these values when there is more than one AP
        ap_thresh = -60
        ap_width = 10
        ap_up_down = 10

        return ap_thresh, ap_width, ap_up_down
    
    else:

        # calculate threshold
        smooth_v = signal.savgol_filter(np.diff(v_np), 101, 3)
        if len(np.where(smooth_v/np.diff(t_np) > 20)[0]) > 0:
            ap_thresh_idx = np.where(smooth_v/np.diff(t_np) > 20)[0][0]+1
            ap_thresh = v_np[ap_thresh_idx]
            # upstroke/downstroke
            # find bottom of AP
            start_ap = int(ap[0])
            stop_ap = np.argmin(np.abs(t_np-(t_np[ap[0]]+5)))
            elbow_p = int(np.argmin(v_np[start_ap:stop_ap]) + start_ap)

            upstroke = np.mean(np.gradient(v_np[ap_thresh_idx:int(ap[0])]))
            downstroke = np.mean(np.gradient(v_np[int(ap[0]):elbow_p]))

            ap_up_down = np.abs(upstroke/downstroke)
        else:
            ap_thresh = 0
            ap_up_down = 100

        # width
        ap_width = properties['widths'][0]*h.dt

        return ap_thresh, ap_width, ap_up_down

def sag_membrane_time_constant(params):

    """Sag ratio & Membrane time constant"""

     # neuron imports
    import neuron
    from neuron import h
    from neuron.units import ms, mV

    # load sections
    sections, mech_class, cell_tree = get_sections_load_mechs(params)

    # load allen data to get stim amplitudes
    _, l2_allen, l3_allen = get_targets()
    sag_stim_amp_l2 = l2_allen['sag_stim_amp'].dropna()
    sag_stim_amp_l2  = np.concatenate(sag_stim_amp_l2.values)

    sag_stim_amp_l3 = l3_allen['sag_stim_amp'].dropna()
    sag_stim_amp_l3  = np.concatenate(sag_stim_amp_l3.values)

    sag_stim_amp = np.unique(np.concatenate((sag_stim_amp_l2, sag_stim_amp_l3)))

    try:
        steady_state = np.zeros_like(sag_stim_amp)
        min_membr_pot = np.zeros_like(sag_stim_amp)
        tau_ms = np.zeros_like(sag_stim_amp)
        for i,amp in enumerate(sag_stim_amp):

            l2pyr= Cell(sections, cell_tree, cell_type='L2_pyramidal', gid=0)
            l2pyr.build()
            l2pyr = mech_class.add_mechs(l2pyr, params)

            stim = h.IClamp(l2pyr._nrn_sections['soma'](0.5))
            stim.delay = 50
            stim.dur = 100
            stim.amp = amp
            clamp_current = h.Vector().record(stim._ref_i)
            t = h.Vector().record(h._ref_t)
            v = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)

            h.celsius = 37
            h.finitialize()
            h.continuerun(200 * ms)

            v_np = v.as_numpy()
            t_np = t.as_numpy()

            # Sag ratio
            steady_idx = np.argmin(np.abs(t-((stim.delay+stim.dur)*0.9)))
            steady_state[i] = v_np[steady_idx]
            min_membr_pot[i] = np.min(v_np)

            # Membrane time constant

            start_idx = np.where(t_np > stim.delay)[0][0]
            end_idx = np.where(t_np > stim.delay + stim.dur*0.9)[0][0]      # steady state, see https://pmc.ncbi.nlm.nih.gov/articles/PMC2775376/

            t_rise = t_np[start_idx:end_idx] - stim.delay # Time starting at 0
            v_rise = v_np[start_idx:end_idx]

            delta_v = np.mean(v_rise) - np.mean(v_np[:start_idx])
            # Fit the exponential function to the data
            popt, pcov = curve_fit(_exp_func, t_rise, v_rise, p0=[10, abs(delta_v), v_rise[0]])
            tau_ms[i] = popt[0] # This is the fitted time constant in 
        
        sag_ratio = np.mean((min_membr_pot-steady_state)/min_membr_pot)
        avg_tau_ms = np.mean(tau_ms)
        return sag_ratio, avg_tau_ms

    except Exception as e:
        print(f"mebrane constant error: {str(e)}", flush=True)
        return 0, 0

def attenuation_from_section(ref_sect, params, amp, stim_dur, stim_l):
    """bAP and hyperpolarization attenuation"""

    import neuron
    from neuron import h
    from neuron.units import ms, mV

    section_names = ['soma', 'apical_trunk', 'apical_1', 'apical_tuft']
    amp_ref = []
    amp_dend = []

    distance = []
    i = 0

    sections, mech_class, cell_tree = get_sections_load_mechs(params)
    l2pyr= Cell(sections, cell_tree, cell_type='L2_pyramidal', gid=0)
    l2pyr.build()
    l2pyr = mech_class.add_mechs(l2pyr, params)
    h.distance(sec = l2pyr._nrn_sections[ref_sect])
    
    
    stim = h.IClamp(l2pyr._nrn_sections[ref_sect](0.5))
    stim.delay = 10
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

                idx = np.argmin(np.abs(t_np - 10)) 
                amp_ref.append(np.max(r_filt) - r_filt[idx])
                
                # penalize if dendritic spike is higher in amplitude than soma, or if depolarization is locked out
                if np.max(d_filt) > np.max(r_filt) or np.mean(d_filt[int(stim_l*(1/h.dt)*0.7):]) > -30:
                    amp_dend.append(100000)
                    upstroke.append(100000)
                else:
                    # upstroke per ms (Gidon et al., 2020 Fig S5 A)
                    upstroke.append(np.max(np.diff(d_filt))/np.diff(t_np)[0])  
                    amp_dend.append(np.max(d_filt) - d_filt[idx])

            elif amp     < 0:
                start_idx = np.argmin(np.abs(t_np - (stim.delay+stim.dur*0.8))) # Gidon 2020
                end_idx = np.argmin(np.abs(t_np - (stim.delay+stim.dur*0.99))) # Gooch, 2022

                amp_ref.append(np.mean(r_filt[start_idx:end_idx]))
                amp_dend.append(np.mean(d_filt[start_idx:end_idx]))

            i += 1

                # calculate attenuation
    atten = np.array(amp_dend) / np.array(amp_ref) * 100

    return atten, distance, upstroke

def get_all_attenuation(params):

    spike_amp, elec_sep, upstroke = attenuation_from_section('soma', params, amp=2, stim_dur=5, stim_l=50)
    try:
        popt, pcov = curve_fit(_exp_decay, np.array(elec_sep), np.array(spike_amp), p0=[100, spike_amp[0] ])
        lambda_spike_amp = popt[0]
    except Exception as e:
        print(f"AP decay error: {str(e)}", flush=True)
        lambda_spike_amp = -100000

    try: 
        popt, pcov = curve_fit(_exp_decay, np.array(elec_sep), np.array(upstroke), p0=[100, upstroke[0] ])
        lambda_upstroke = popt[0]
    except Exception as e:
        print(f"AP upstroke error: {str(e)}", flush=True)
        lambda_upstroke = -100000

    # This just kills everything, take out
    # hyperpol_amp, elec_sep, _ = attenuation_from_section('soma', params, amp=-0.3, stim_dur=100, stim_l=120)
    # try:   
    #     popt, pcov = curve_fit(_exp_decay, np.array(elec_sep), np.array(hyperpol_amp), p0=[100, hyperpol_amp[0]])
    #     lambda_hyperpol_amp = popt[0]
    # except Exception as e:
    #     print(f"hyperpol attenuation error: {str(e)}", flush=True)
    #     lambda_hyperpol_amp = -100000


    return lambda_spike_amp, lambda_upstroke #, lambda_hyperpol_amp

def isi_FI_slope(params):

    import neuron
    from neuron import h
    from neuron.units import ms, mV

    sections, mech_class, cell_tree = get_sections_load_mechs(params)

    # get the amplitude used to determine ISIs
    soma_dend_atten, l2_allen, l3_allen = get_targets()
    amp_isi = np.mean(np.concatenate((l2_allen['max_amp_isi'].dropna().values,l3_allen['max_amp_isi'].dropna().values)))

    max_amp_isi = np.max(np.concatenate((l2_allen['max_amp_isi'].dropna().values,l3_allen['max_amp_isi'].dropna().values)))

    l_stim = 1000

    ampl = np.array([amp_isi/3, amp_isi/1.5, amp_isi, max_amp_isi/1.5, max_amp_isi])
    fr = np.zeros_like(ampl)

    for i,amp in enumerate(ampl):

            l2pyr= Cell(sections, cell_tree, cell_type='L2_pyramidal', gid=0)
            l2pyr.build()
            l2pyr = mech_class.add_mechs(l2pyr, params)

            stim = h.IClamp(l2pyr._nrn_sections['soma'](0.5))
            stim.delay = 50
            stim.dur = 1000
            stim.amp = amp
            clamp_current = h.Vector().record(stim._ref_i)
            t = h.Vector().record(h._ref_t)
            v = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)

            h.celsius = 37
            h.dt = 0.01
            h.finitialize()
            h.continuerun(1050 * ms)

            v_np = v.as_numpy()
            t_np = t.as_numpy()

            pk = find_peaks(v_np, height=0)

            fr[i] = len(pk[0])
            if amp == amp_isi:
                try:
                    first_isi = np.diff(t.as_numpy()[pk[0]])[0]
                    mean_isi = np.mean(np.diff(t.as_numpy()[pk[0]]))
                except Exception as e:
                    print(f"ISI error: {str(e)}", flush=True)
                    first_isi = 100000
                    mean_isi = 100000

            # penalize if depolarization locks
            if len(pk) < 4 and np.mean(v_np[int(1050*(1/h.dt)*0.7):]) >= -55:
                penalty = 10000
            # penalize firing rates > 20 Hz
            elif len(pk[0]) > 20:
                penalty = 100
            else:
                penalty = 0
            fr[i] += penalty

    fr_slope = np.max(np.gradient(fr))*0.001 # convert to pA

    return fr_slope, first_isi, mean_isi, penalty

def quick_sanity_check(params_dict):
    """Quick checks before running full simulation"""
    
    # Check if passive parameters are in reasonable ranges
    if not (0.5 <= params_dict['passive_params']['L2Pyr_soma_cm'] <= 3.0):
        return False
    if not (70 <= params_dict['passive_params']['L2Pyr_soma_Ra'] <= 200):
        return False
    
    # Check if reversal potentials are reasonable
    if not (-90 <= params_dict['e_pas'] <= -65):
        return False
    
    return True

# mutation fitness
def mut_fitness(solution):

    soma_dend_atten, l2_allen, l3_allen = get_targets()

    ephys = dict()
    measures = ['RMP', 'upstroke_downstroke', 'ap_width', 'ap_threshold', 'tau', 'sag', 'mean_isi', 'first_isi',  'FR_slope']

    # bring params in the right format
    params_dict = reconstruct_params(solution)  

    # if not quick_sanity_check(params_dict):
    #     return 1e-100  # Very low fitness

    # RMP
    #print("RMP...", flush=True)
    ephys['RMP'] = rmp(params_dict)

    if ephys['RMP'] > -50:
        return 1e-100           # kill early because there is no point if this is too high

    # AP measures
    #print("AP...", flush=True)
    ap_thresh, ap_width, ap_up_down = ap_measures(params_dict)
    ephys['ap_width'], ephys['ap_threshold'], ephys['upstroke_downstroke'] = ap_width, ap_thresh, ap_up_down

    # Membrane time constant
    #print("Membrane time constant...", flush=True)
    sag, tau = sag_membrane_time_constant(params_dict)
    ephys['tau'] = tau
    ephys['sag'] = sag

    # FR slope and ISI
    fi_slope, first_isi, mean_isi, penalty = isi_FI_slope(params_dict)
    ephys['FR_slope'] = fi_slope
    ephys['first_isi'] = first_isi
    ephys['mean_isi'] = mean_isi
    
    # errors for the measures that we have data for
    error_i = np.zeros((len(ephys)+4,))
    weight_i = np.zeros((len(ephys)+4,)) # define tolerance
    for i,m in enumerate(measures):
        target = np.concatenate((l2_allen[m].dropna().values,l3_allen[m].dropna().values))
        m_target = np.mean(target)
        std_target = np.std(target)

        error_i[i] = np.abs(ephys[m] - m_target)/std_target
        if m == 'FR_slope':
            error_i[i] += penalty

        # stay around mean difference in data
        diff_to_mean = target-m_target
        # get distance within quartile
        target_range = np.mean(np.abs(np.quantile(diff_to_mean, [0.25, 0.75])))
        weight_i[i] = 1/target_range

        # this one is super important because it messes with everything
        if m == 'RMP' and ephys[m] > -65:
            weight_i[i] *= 100

    # Attenuation
    #print('Attenuation...', flush=True)
    #lambda_spike_amp, lambda_upstroke, lambda_hyperpol_amp = get_all_attenuation(params_dict)
    lambda_spike_amp, lambda_upstroke = get_all_attenuation(params_dict)
    # get error (divide by value itself to normalize by magnitude)
    error_i[i+1] = np.abs(soma_dend_atten['bAp'] - lambda_spike_amp)/soma_dend_atten['bAp']
    weight_i[i+1] = 1/(soma_dend_atten['bAp']*0.25)
    error_i[i+2] = np.abs(soma_dend_atten['bAp_upstroke'] - lambda_upstroke)/soma_dend_atten['bAp_upstroke']
    weight_i[i+2] = 1/(soma_dend_atten['bAp_upstroke']*0.25)
    # error_i[i+3] = np.abs(soma_dend_atten['hyperpol'] - lambda_hyperpol_amp)/soma_dend_atten['hyperpol']
    # weight_i[i+3] = 1/(soma_dend_atten['hyperpol']*0.25)

    # Input resistance
    inp_res = input_res(params_dict)
    error_i[i+4] = (soma_dend_atten['inp_res']-inp_res)/soma_dend_atten['inp_res']
    weight_i[i+4] = 1/soma_dend_atten['inp_res']*0.25

    # fitness based on error and weight
    fitness = 1/(error_i @ weight_i)

    return fitness

def fitness_function(ga_instance, solution, solution_idx):

    soma_dend_atten, l2_allen, l3_allen = get_targets()

    ephys = dict()
    measures = ['RMP', 'upstroke_downstroke', 'ap_width', 'ap_threshold', 'tau', 'sag', 'mean_isi', 'first_isi',  'FR_slope']

    # bring params in the right format
    params_dict = reconstruct_params(solution)  

    # if not quick_sanity_check(params_dict):
    #     return 1e-100  # Very low fitness

    # RMP
    #print("RMP...", flush=True)
    ephys['RMP'] = rmp(params_dict)

    # AP measures
    #print("AP...", flush=True)
    ap_thresh, ap_width, ap_up_down = ap_measures(params_dict)
    ephys['ap_width'], ephys['ap_threshold'], ephys['upstroke_downstroke'] = ap_width, ap_thresh, ap_up_down

    # Membrane time constant
    #print("Membrane time constant...", flush=True)
    sag, tau = sag_membrane_time_constant(params_dict)
    ephys['tau'] = tau
    ephys['sag'] = sag

    # FR slope and ISI
    fi_slope, first_isi, mean_isi, penalty = isi_FI_slope(params_dict)
    ephys['FR_slope'] = fi_slope
    ephys['first_isi'] = first_isi
    ephys['mean_isi'] = mean_isi
    
    # errors for the measures that we have data for
    error_i = np.zeros((len(ephys)+4,))
    weight_i = np.zeros((len(ephys)+4,)) # define tolerance
    for i,m in enumerate(measures):
        target = np.concatenate((l2_allen[m].dropna().values,l3_allen[m].dropna().values))
        m_target = np.mean(target)
        std_target = np.std(target)

        error_i[i] = np.abs(ephys[m] - m_target)/std_target
        if m == 'FR_slope':
            error_i[i] += penalty

        # stay around mean difference in data
        diff_to_mean = target-m_target
        # get distance within quartile
        target_range = np.mean(np.abs(np.quantile(diff_to_mean, [0.25, 0.75])))
        weight_i[i] = 1/target_range

        # this one is super important because it messes with everything
        if m == 'RMP' and ephys[m] > -65:
            weight_i[i] *= 10

    # Attenuation
    #print('Attenuation...', flush=True)
    #lambda_spike_amp, lambda_upstroke, lambda_hyperpol_amp = get_all_attenuation(params_dict)
    lambda_spike_amp, lambda_upstroke = get_all_attenuation(params_dict)
    # get error (divide by value itself to normalize by magnitude)
    error_i[i+1] = np.abs(soma_dend_atten['bAp'] - lambda_spike_amp)/soma_dend_atten['bAp']
    weight_i[i+1] = 1/(soma_dend_atten['bAp']*0.25)
    error_i[i+2] = np.abs(soma_dend_atten['bAp_upstroke'] - lambda_upstroke)/soma_dend_atten['bAp_upstroke']
    weight_i[i+2] = 1/(soma_dend_atten['bAp_upstroke']*0.25)
    # error_i[i+3] = np.abs(soma_dend_atten['hyperpol'] - lambda_hyperpol_amp)/soma_dend_atten['hyperpol']
    # weight_i[i+3] = 1/(soma_dend_atten['hyperpol']*0.25)

    # Input resistance
    inp_res = input_res(params_dict)
    error_i[i+4] = (soma_dend_atten['inp_res']-inp_res)/soma_dend_atten['inp_res']
    weight_i[i+4] = 1/soma_dend_atten['inp_res']*0.25

    # fitness based on error and weight
    fitness = 1/(error_i @ weight_i)

    return fitness

def on_fitness(ga_instance, population_fitness):
    print("on_fitness()", flush=True)

def on_parents(ga_instance, selected_parents):
    print("on_parents()", flush=True)

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()", flush=True)

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()", flush=True)
    
    # try:
    #     # Apply local search only every 10 generations
    #     if ga_instance.generations_completed % 10 == 0:
    #         print("Applying Optuna-guided mutation", flush=True)
            
    #         # Your existing mutation code here
    #         fitness_values = [mut_fitness(sol) for sol in offspring_mutation]
    #         best_offsprings_idx = np.argsort(-np.array(fitness_values))[:5]
            
    #         for idx in best_offsprings_idx:
    #             # Your Optuna code here
    #             pass
                
    #     return offspring_mutation
        
    # except Exception as e:
    #     print(f"Error in on_mutation: {str(e)}", flush=True)
    #     import traceback
    #     traceback.print_exc()
    #     return offspring_mutation  # Always return the offspring, even on error

# def on_mutation(ga_instance, offspring_mutation):

#     print("on_mutation()", flush=True)

#     return offspring_mutation

#     # # Pick random offsprings
#     # if ga_instance.generations_completed % 10 == 0:  # every 10 generations

#     #     fitness_values = [mut_fitness(sol) for sol in offspring_mutation]

#     #     # take the best 5 offsprings and mutate them
#     #     best_offsprings_idx = np.argsort(-np.array(fitness_values))[:5]              # sort the negative values because argsort only does ascending

#     #     for idx in best_offsprings_idx:

#     #         offspring = np.array(offspring_mutation[idx])
#     #         param_space = np.zeros((len(offspring),2))
#     #         param_space[:,0] = offspring*0.8
#     #         param_space[:,1] = offspring*1.2

#     #         objective_fun = partial(objective, ga_instance, param_space, idx)

#     #         # Each process gets its own study instance but shares the same storage
#     #         study = optuna.create_study(direction='maximize', load_if_exists=False)
            
#     #         # Run trials
#     #         study.optimize(objective_fun, n_trials=10)

#     #         offspring_mutation[idx,:] = list(study.best_params.values())
            
def objective(ga_instance, param_space, solution_idx, trial):

    print(f"Running trial {trial.number=}", flush=True)
                
    param_names = ['NaTs2_t_32d_soma','SKv3_1_soma', 'Nap_Et2_soma', 'Ca_HVA_soma', 'Ca_LVAst_soma', 'pas_soma', 'Ih_soma', 'SK_E2_soma', 'Im_soma', 'NaTs2_t_32d_basal', 'SKv3_1_basal', 'Ih_basal', 'pas_basal',
                        'NaTa_t_32d_apical', 'SKv3_1_apical','Ca_HVA_apical', 'Ca_LVAst_apical', 'pas_apical', 'SK_E2_apical', 'Ih_apical', 'Im_apical','decay', 'gamma',
                        'e_pas', 'cm', 'Ra']
    proposed_vals = []
    for key, val in zip(param_names, param_space):

        if key == 'e_pas':
            proposed_vals.append(trial.suggest_float(key, val[1], val[0]))
        else:
            proposed_vals.append(trial.suggest_float(key, val[0], val[1]))  

    fit = fitness_function(ga_instance, proposed_vals, solution_idx)  

    return fit

def on_generation(ga_instance):
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        print(f'Generation : {ga_instance.generations_completed} at {now.date()} {current_time}', flush=True)

        np.save(f'{pth_rslts}/last_gen_parents.npy', ga_instance.last_generation_parents)
        np.save(f'{pth_rslts}/current_pop.npy', ga_instance.population)
        np.save(f'{pth_rslts}/last_gen_parents_fit.npy',
                ga_instance.last_generation_fitness[ga_instance.last_generation_parents_indices])
        fitness_list.append(ga_instance.last_generation_fitness)
        np.save(f'{pth_rslts}/generation_fitness.npy', np.array(fitness_list))
        parents_list.append(ga_instance.last_generation_parents)
        np.save(f'{pth_rslts}/all_parents.npy', np.array(parents_list))

        ga_instance.save(f'{pth_rslts}/pygad_inst')

def on_stop(ga_instance, last_population_fitness):
    
    print("Done")
    best_sol = ga_instance.best_solution()

    with open(f'{pth_rslts}/best_sol.pkl', 'wb') as file:
        pickle.dump(best_sol, file)

def informed_initialization(gene_space, population_size, initial_solution):
    """Create initial population with informed sampling"""
    # Create population array
    population = np.empty((population_size-1, len(gene_space)))
    
    for i, (gene, initial_val) in enumerate(zip(gene_space, initial_solution)):
        low, high = gene

        #if high/low > 100:
        # Sample with normal distribution around mean
        std = (high - low)  # Standard deviation is 1/3 of the range
        samples = np.random.uniform(low, high, population_size-1)
        
        # # conductances
        # if np.abs(low) < 1:
        #         samples = np.clip(samples, low, np.max(samples)) # clip lower level at 0
        # # other clip in range
        # else: 
        #         samples = np.clip(samples, low, high)

        # if high == 1/9000:
        #         samples = np.random.uniform(low, high, population_size-1)

        population[:,i] = samples

    return population

def main():
    
    num_gen = 300
    population_size = 301
    num_parents_mating = int(population_size*0.2)  


    {'soma': {'NaTs2_t_32d': 20_400e-4*2,
                                'SKv3_1': 6_930e-4*1.5,
                                'Nap_Et2': 1e-4/2*10,
                                'Ca_HVA': 9.92e-04,
                                'Ca_LVAst': 34.3e-4*0.1,
                                'pas': 1.0 / 38331,
                                'Ih': 0.000080*10,
                                'SK_E2': 5e-02,
                                'Im': 8e-3*0.001},
                        'basal': {'NaTs2_t_32d': 0.008009,
                                  'SKv3_1': 0.000513,
                                  'Ih': 0.00008*10,
                                  'pas':  1.0 / 30000.0},
                        'dend': {'NaTa_t_32d': 0.0007545577135222227,
                                'SKv3_1': 0.008911542253622935,
                                'Ca_HVA': 0.0010229989264147366,
                                'Ca_LVAst': 0.0010793073879112331,
                                'pas': 1.0 /29062,
                                'SK_E2': 5.183204947473679e-05,
                                'Ih': 0.000256109816290954,
                                'Im': 0.004888309098371485,
                                'CaDynamics_E2': {'decay': 132.59067310238868, 'gamma': 0.0025792496905644342}},
                        'e_pas': -75,
                        'passive_params': {'cm': 2,
                                           'Ra': 200}}

    gene_space = [[1.5, 6],
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

    initial_params = _get_start_params()
    initial_solution = flatten_params(initial_params)
    

    # generate start values
    initial_population = informed_initialization(gene_space, population_size, initial_solution)

    initial_solution = np.array(initial_solution).reshape(1, -1)
    initial_population = np.concatenate((initial_solution, initial_population), axis=0)

    global fitness_list
    global parents_list

    fitness_list = []
    parents_list = []

    
    ga_instance = pygad.GA(num_generations=num_gen,
                    num_parents_mating=num_parents_mating,
                    sol_per_pop = population_size,
                    allow_duplicate_genes=False,  # Prevent duplicate solutions
                    fitness_func=fitness_function,
                    num_genes=len(gene_space),
                    initial_population=initial_population,
                    crossover_type="uniform",  # Uniform crossover may keep diversity
                    crossover_probability=0.5,  # Crossover happens frequenctly but not all the time
                    keep_parents=2,  # Keep fewer parents to increase diversity
                    parent_selection_type="rws",  # roulette wheel selection
                    mutation_type='random',  
                    mutation_probability=0.3,  # Higher mutation probability
                    mutation_by_replacement=True,  # Ensure mutations stay within bounds
                    suppress_warnings = True,  # Reduce output noise
                    gene_space=gene_space,
                    on_fitness=on_fitness,
                    on_parents=on_parents,
                    on_crossover=on_crossover,
                    on_mutation=on_mutation,
                    on_generation=on_generation,
                    on_stop=on_stop,
                    parallel_processing=["process", 43])

    ga_instance.run()

    ga_instance.save(f'{pth_rslts}/pygad_inst')

if __name__ == "__main__":
    main()