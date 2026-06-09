""""Collection of utils used for both L5 ET and L2/3"""
import neuron
from neuron import h
from neuron.units import ms
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from l23_plot_utils import get_l2pyr
from l5_plot_utils import get_l5pyr
import matplotlib.pyplot as plt

import os.path as op
pth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning'

pth_epsp_mod = op.join(pth, 'new_cells_dev')            # file with EPSP point process
neuron.load_mechanisms(pth_epsp_mod)
def exp_func(t, tau, A, C):
    return A * (1 - np.exp(-t / tau)) + C

def exp_decay(x, lambda_val, A):
    return A * np.exp(-x / lambda_val)

def attenuation_from_section(model, ref_sect, section_names, amp, stim_dur, ax, cols_dist, epsp=False):

    if epsp:
        syn_inp = h.epsp(model._nrn_sections[ref_sect](0.9))
        syn_inp.onset = 20
        syn_inp.imax = 0.2
    else:
        if ref_sect == 'apical_tuft':
            stim = h.IClamp(model._nrn_sections[ref_sect](0.9))
        else:
            stim = h.IClamp(model._nrn_sections[ref_sect](0.5))
        stim.delay = 20
        stim.dur = stim_dur
        stim.amp = amp

    t = h.Vector().record(h._ref_t)
    
    ref_voltg = h.Vector().record(model._nrn_sections[ref_sect](0.9)._ref_v)
    
    #section_oi = section_names.copy()
    if ref_sect == 'apical_tuft':
        section_oi = section_names[::-1]
    else:
        section_oi = [section for section in section_names if section != ref_sect]

    upstroke = []
    dend_voltg = []
    
    distance = []
    
    for s, section in enumerate(section_oi):
        print(f'ref section: {ref_sect}, recording from section: {section}')
        for sg, seg in enumerate(model._nrn_sections[section]):

            dend_voltg.append(h.Vector().record(model._nrn_sections[section](seg.x)._ref_v))

            distance.append(h.distance(model._nrn_sections[ref_sect](0.9), model._nrn_sections[section](seg.x)))

    h.celsius = 37
    h.finitialize()
    h.continuerun(1100 * ms)

    ax.plot(t, ref_voltg, color=cols_dist[0])
    
    t_np = t.as_numpy()
    if epsp:
        bsl_idx = np.where(t_np <= syn_inp.onset)[0][-1]
    else:
        bsl_idx = np.where(t_np <= stim.delay)[0][-1]
    amp_ref = []
    amp_dend = []
    i = 0
    # plot_min = 1000
    # plot_max = -1000
    for s, section in enumerate(section_oi):
        for sg, seg in enumerate(model._nrn_sections[section]):
            # low-pass filter 5KHz like in Gidon et al.,
            sfreq = 1000/np.diff(t.as_numpy())[0]
            b, a = signal.butter(0, 5000, btype='low', analog=False, fs=sfreq)

            v = dend_voltg[i].as_numpy()
            d_filt = v #signal.lfilter(b, a, v)
            r_filt = ref_voltg.as_numpy() #signal.lfilter(b, a, ref_voltg.as_numpy())
            

            if amp > 0:  
                # upstroke per ms (Gidon et al., 2020 Fig S5 A)
                upstroke.append(np.max(np.diff(d_filt))/np.diff(t_np)[0])     

                amp_ref.append(np.max(r_filt) - np.mean(r_filt[:bsl_idx]))
                amp_dend.append(np.max(d_filt) - np.mean(d_filt[:bsl_idx]))

            elif amp     < 0:
                pre_stim_soma = r_filt[np.where(t_np < stim.delay)[0][-1]]
                pre_stim_dend = d_filt[np.where(t_np < stim.delay)[0][-1]]

                start_idx = np.argmin(np.abs(t_np - (stim.delay+800))) # Gidon 2020
                end_idx = np.argmin(np.abs(t_np - (stim.delay+1000))) # Gooch, 2022

                amp_ref.append(np.abs((np.mean(r_filt[start_idx:end_idx])-pre_stim_soma)))
                amp_dend.append(np.abs((np.mean(d_filt[start_idx:end_idx])-pre_stim_dend)))

            
            ax.plot(t_np,v, color=cols_dist[i+1])
            i += 1
            # plot_min = min(np.round(v.min()), plot_min)
            # plot_max = max(np.round(v.max()), plot_max)

    atten = np.array(amp_dend)/np.array(amp_ref)*100

    # ax.set_ylim([plot_min, plot_max])
    # ax.set_yticks([plot_min, plot_max])
    idx = np.argsort(distance)
    atten = atten[idx]
    distance = np.array(distance)[idx]

    if amp>0:
        upstroke = np.array(upstroke)[idx]

    return atten, distance, upstroke, ax

## Plot attenuation

def plot_attenuation(cols_dist, model_name='l2pyr', ref_sect='soma', epsp=False, amp_hyperpol=-2, amp_spike=2):
    
    fig, ax = plt.subplots(1,5, figsize=(17, 2.5), constrained_layout=True)

    if model_name == 'l2pyr':
        model = get_l2pyr()
        section_names = ['soma', 'apical_trunk', 'apical_1', 'apical_tuft']
    elif model_name == 'l5pyr':
        model = get_l5pyr()
        section_names = ['soma', 'apical_trunk', 'apical_1', 'apical_2', 'apical_tuft']
    else:
        raise ValueError('model must be either "l2pyr" or "l5pyr"')
    # hyperpolarization attenuation
    hyperpol_atten, elec_sep, _, ax[0] = attenuation_from_section(model, ref_sect, section_names, amp=-2, stim_dur=1000, ax=ax[0], cols_dist=cols_dist, epsp=False)
    popt, pcov = curve_fit(exp_decay, np.array(elec_sep), np.array(hyperpol_atten), p0=[100, hyperpol_atten[0] ])
    lambda_hyperpol = popt[0]

    ax[1].plot(elec_sep, hyperpol_atten, 'o', color='w', markeredgecolor='k', markersize=7)
    ax[1].set_xlim(np.floor(elec_sep[0]), np.round(elec_sep[-1]+100, -2))
    ax[1].set_xticks(np.linspace(0, np.round(elec_sep[-1]+100, -2), 3))
    ax[1].set_xlabel(f'distance from {ref_sect} ($\mu m$)')
    ax[1].set_ylabel('attenuation (%)')
    ax[1].set_ylim(0, 101)
    ax[1].plot(elec_sep, exp_decay(np.array(elec_sep), *popt), color='k')
    ax[1].text(200, 70, f'{lambda_hyperpol:.2f}') # Gidon 2020 475
    # spike attenuation
    if model_name == 'l2pyr':
        model = get_l2pyr()
    elif model_name == 'l5pyr':
        model = get_l5pyr()
    else:
        raise ValueError('model must be either "l2pyr" or "l5pyr"')
    spike_amp, _, upstroke, ax[2] = attenuation_from_section(model, ref_sect, section_names, amp=2, stim_dur=5,ax=ax[2], cols_dist=cols_dist, epsp=epsp)

    ax[3].plot(elec_sep, spike_amp, 'o', color='w', markeredgecolor='k', markersize=7)
    ax[3].set_xlim(np.floor(elec_sep[0]), np.round(elec_sep[-1]+100, -2))
    ax[3].set_xticks(np.linspace(0, np.round(elec_sep[-1]+100, -2), 3))
    popt, pcov = curve_fit(exp_decay, np.array(elec_sep), np.array(spike_amp), p0=[100, spike_amp[0] ])
    lambda_spike_amp = popt[0]
    ax[3].plot(elec_sep, exp_decay(np.array(elec_sep), *popt), color='k')
    ax[3].text(100, spike_amp.max(), f'{lambda_spike_amp:.2f}') # Gidon 2020, 290

    ax[3].set_xlabel(f'distance from {ref_sect} ($\mu m$)')
    ax[3].set_ylabel('attenuation (%)')

    ax[2].set_xlim([0, 60])
    ax[3].set_yticks(np.arange(0, 101, 25))


    ax[4].plot(elec_sep, upstroke, 'o', color='w', markeredgecolor='k', markersize=7)
    ax[4].set_xlim(np.floor(elec_sep[0]), np.round(elec_sep[-1]+100, -2))
    ax[4].set_xticks(np.linspace(0, np.round(elec_sep[-1]+100, -2), 3))
    popt, pcov = curve_fit(exp_decay, np.array(elec_sep), np.array(upstroke), p0=[100, upstroke[0] ])
    ax[4].plot(elec_sep, exp_decay(np.array(elec_sep), *popt), color='k')
    lambda_upstroke = popt[0]
    ax[4].set_ylabel('bAP upstroke (mV/ms)')
    ax[4].text(100, upstroke.max(), f'{lambda_upstroke:.2f}')

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    plt.show(block=True)
        
    return fig