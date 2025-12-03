"""L2/3 plot utils """

import neuron
from neuron import h
import matplotlib.pyplot as plt
import matplotlib
from neuron.units import ms, mV
import numpy as np
pth_mod = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/local_hnn/hnn_core/mod'
figpth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/new_cells_dev/figures/l23_tuning/'
import os
os.makedirs(figpth, exist_ok=True)
plt.rcParams['svg.fonttype'] = 'none'

import sys
sys.path.append('/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/new_cells_dev/utils')
from cell_builder_utils import l2_params, _get_dends, _get_pyr_soma, Cell
from l23_mechs_utils import Mechs
from functools import partial
from l5_mechs_utils import linear_gbar, Ih_condu_hay
from hnn_core.viz import plot_cell_morphology
from matplotlib import colormaps
from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def exp_func(t, tau, A, C):
    return A * (1 - np.exp(-t / tau)) + C

def exp_decay(x, lambda_val, A):
    return A * np.exp(-x / lambda_val)

def get_colormap(cmap_name='inferno'):

    cmap = colormaps['inferno']
    cols_dist = cmap.colors[0:len(cmap.colors):17]
    cm_neurons = np.array([[218,166,156], [164,30,79], [119,161,187],[92,114,183]])/255
    cols_apcl = np.array(([27, 158, 119], [217,95,2], [117,112,179], [231,41,138], [102,166,30]))/255
    cols_basal = np.array(([56,108,176], [102,166,30], [231,41,138], [231,41,138]))/255

    """ Get colormap for plotting """
    l2pyr = get_l2pyr()
    distance = []
    h.distance(sec = l2pyr._nrn_sections['soma'])
    section_names = ['soma', 'apical_trunk','apical_1', 'apical_tuft']

    for section in section_names:
        for seg in l2pyr._nrn_sections[section]:
            distance.append(h.distance(seg))

    im = plt.pcolormesh(np.tile(np.array(distance),(2,1)), cmap=cmap_name, vmin=0, vmax=2000)
    plt.clf()

    return im, cols_dist, cm_neurons, cols_apcl, cols_basal

def get_model_params():

    return {'soma': {'NaTs2_t_32d': 20_400e-4*1.2,
                                'SKv3_1': 6_930e-4*1.5,
                                'Nap_Et2': 1e-4/2*0,
                                'Ca_HVA': 9.92e-04,
                                'Ca_LVAst': 34.3e-4,
                                
                                'SK_E2': 3.e-05,
                                'pas': 1.0 / 37000,
                                'Ih': 0.000080*2,#/2,
                                'Im': 8e-3*0.05,
                                'CaDynamics_E2': {'decay': 90, 'gamma': 0.00053300}},
                        'basal': {'NaTs2_t_32d': 0.008009,#partial(linear_gbar,0.008009/2,0,255),
                                  'SKv3_1': 0.000513,#partial(linear_gbar,0,6_930e-4*0.1,255),
                                  'Ih': 0.00008*2,#/2,
                                  'pas':  1.0 / 37000.0},
                        'dend': {'NaTa_t_32d': 0.008*0.01,
                                'SKv3_1': 0.003*4,
                                'Ca_HVA': partial(linear_gbar, 0.00001, 0.002, 200, hotzone=[200, 400], hotzone_factor=4),#0.0013791999171895088/10,
                                'Ca_LVAst': partial(linear_gbar, 0.0000001, 0.001, 200, hotzone=[200, 400], hotzone_factor=4),
                                'pas': 1.0 /37000,
                                'SK_E2': partial(linear_gbar, 3.e-06, 3.e-03, 200, hotzone=[200, 400], hotzone_factor=10), #3.547577318022417e-05,
                                'Ih': partial(Ih_condu_hay, 0.00008),
                                'Im': 0.0009*0.0001,
                                'CaDynamics_E2': {'decay': 70, 'gamma': 0.00053300*2.5}},
                        'e_pas': -75}

def get_l2pyr(v_init={'soma': -73.91534035708615,
                    'basal_1': -73.93352687563397,
                    'basal_2': -73.98646383934121,
                    'basal_3': -73.98646383934121,
                    'apical_oblique': -73.9142120929283,
                    'apical_trunk': -73.88877758950669,
                    'apical_1': -73.64560269252749,
                    'apical_tuft': -73.27793049058006}):
    
    model_params = get_model_params()
    
    # get paramaters
    section_params, end_pts, cell_tree = l2_params()

    # build sections
    section_names = list(end_pts.keys())


    sections = _get_dends(section_params, 'L2Pyr', section_names, end_pts, v_init=v_init)
    sections['soma'] = _get_pyr_soma(section_params, 'L2Pyr', end_pts['soma'], v_init=v_init['soma'])

    mech_class = Mechs(pth_mod)

    l2pyr= Cell(sections, cell_tree, cell_type = 'L2_pyramidal', gid=0)
    l2pyr.build()
    l2pyr = mech_class.add_mechs(l2pyr, model_params)

    return l2pyr

def plot_conductances():

    _, _, _, cols_apcl, cols_basal = get_colormap()
   
    l2pyr = get_l2pyr()
    distance_apical = []
    gbar_Ih_apical = []
    gbar_Ca_HVA_apical = []
    gbar_Ca_LVA_apical = []
    gbar_SK_E2_apical = []

    apical_sections=['apical_trunk', 'apical_1', 'apical_tuft']
    h.distance(sec = l2pyr._nrn_sections['soma'])
    for section in apical_sections:
        for seg in l2pyr._nrn_sections[section]:
            distance_apical.append(h.distance(seg.x, sec=seg.sec))
            gbar_Ih_apical.append(seg.gbar_Ih)
            gbar_Ca_HVA_apical.append(seg.gbar_Ca_HVA)
            gbar_Ca_LVA_apical.append(seg.gbar_Ca_LVAst)
            gbar_SK_E2_apical.append(seg.gbar_SK_E2)

    section_names = ['apical_tuft', 'apical_1', 'apical_trunk', 'basal_1', 'basal_2', 'basal_3', 'apical_oblique', 'soma']
    inp_res_apcl = []
    inp_res_basal = []
    inp_res_soma = []
    inp_res_oblq = []
    distance_basal = []
    distance_oblq = []
    distance_apcl = []

    for s, section in enumerate(section_names):
        for i, seg in enumerate(l2pyr._nrn_sections[section]):

            impedance = h.Impedance()
            h.celsius = 37
            h.finitialize()
            impedance.loc(seg.x, sec=l2pyr._nrn_sections[section])
            impedance.compute(0)

            if 'basal' in section:
                inp_res_basal.append(impedance.input(l2pyr._nrn_sections[section](seg.x)))
                distance_basal.append(h.distance(seg))
            elif section == 'apical_oblique':
                inp_res_oblq.append(impedance.input(l2pyr._nrn_sections[section](seg.x)))
                distance_oblq.append(h.distance(seg))
            elif 'apical' in section:
                inp_res_apcl.append(impedance.input(l2pyr._nrn_sections[section](seg.x)))
                distance_apcl.append(h.distance(seg))

            elif section == 'soma':
                inp_res_soma.append(impedance.input(l2pyr._nrn_sections[section](seg.x)))


    # Fit membrane time constant

    tau_m_apcl_cf = []
    tau_m_basal_cf = []
    tau_m_soma_cf = []
    tau_m_oblq_cf = []

    for s, section in enumerate(section_names):
        for i, seg in enumerate(l2pyr._nrn_sections[section]):

            l2pyr= get_l2pyr()
            h.distance(sec = l2pyr._nrn_sections['soma'])

            stim = h.IClamp(l2pyr._nrn_sections['soma'](0.5))
            stim.delay = 50
            stim.dur = 1000
            stim.amp = -0.15

            voltg = h.Vector().record(l2pyr._nrn_sections[section](0.5)._ref_v)

            t = h.Vector().record(h._ref_t)

            h.celsius = 37
            h.finitialize()
            h.continuerun(1150 * ms)

            v_np = voltg.as_numpy()
            t_np = t.as_numpy()
            start_idx = np.where(t_np > stim.delay)[0][0]
            end_idx = np.where(t_np > stim.delay + stim.dur*0.9)[0][0]      # steady state, see https://pmc.ncbi.nlm.nih.gov/articles/PMC2775376/

            t_rise = t_np[start_idx:end_idx] - stim.delay # Time starting at 0
            v_rise = v_np[start_idx:end_idx]

            delta_v = np.mean(v_rise) - np.mean(v_np[:start_idx])
            # Fit the exponential function to the data
            popt, pcov = curve_fit(exp_func, t_rise, v_rise, p0=[10, abs(delta_v), v_rise[0]])
            tau_m_fitted = popt[0] # This is the fitted time constant in ms

            
            if section == 'soma':
                tau_m_soma_cf.append(tau_m_fitted)
            elif 'basal' in section:
                tau_m_basal_cf.append(tau_m_fitted)
            elif section == 'apical_oblique':
                tau_m_oblq_cf.append(tau_m_fitted)
            elif 'apical' in section:
                tau_m_apcl_cf.append(tau_m_fitted)

    fig, ax = plt.subplots(3,2,figsize=(8,6), constrained_layout=True)
    ax = ax.flatten()
    ax[0].plot(distance_apical, gbar_Ih_apical, color='k')
    ax[0].set_ylabel('$gbar_{Ih} (S/cm^2)$')
    ax[0].set_ylim([0, 0.001])
    ax[0].set_yticks(np.arange(0, 0.0011, 0.0005))
    ax[1].plot(distance_apical, gbar_Ca_HVA_apical, color='k')
    ax[1].set_ylabel('$gbar_{Ca_{HVA}} (S/cm^2)$')
    ax[1].set_ylim([0, 0.009])
    ax[1].set_yticks(np.linspace(0, 0.009, 3))
    ax[2].plot(distance_apical, gbar_Ca_LVA_apical, color='k')
    ax[2].set_ylabel('$gbar_{Ca_{LVA}} (S/cm^2)$')
    ax[2].set_ylim([0, 0.005])
    ax[2].set_yticks(np.arange(0, 0.0051, 0.0025))
    ax[3].plot(distance_apical, gbar_SK_E2_apical, color='k')
    ax[3].set_ylabel('$gbar_{SK-E2} (S/cm^2)$')
    ax[3].set_ylim([0, 0.04])
    ax[3].set_yticks([0, 0.02, 0.04])
    ax[3].set_xlabel('distance to soma ($\mu m$)')
    ax[2].set_xlabel('distance to soma ($\mu m$)')
    ax[4].plot(0, inp_res_soma, 'o', color='k')
    ax[4].plot(distance_basal, inp_res_basal, 'o', color=cols_basal[0], label='basal')
    ax[4].plot(distance_oblq, inp_res_oblq, 'o', color=cols_basal[2], label='apcl oblqu')
    ax[4].plot(distance_apcl, inp_res_apcl, 'o', color=cols_apcl[0], label='apical')
    ax[4].set_ylabel('input resistance') # (M$\Omega$)')
    ax[4].set_xlabel('distance to soma')# ($\mu m$)')
    ax[5].plot(0, tau_m_soma_cf, 'o', color='k', label='soma')
    ax[5].plot(distance_basal, tau_m_basal_cf, 'o', color=cols_basal[0], label='basal')
    ax[5].plot(distance_oblq,  tau_m_oblq_cf, 'o', color=cols_basal[2], label='apcl oblqu')
    ax[5].plot(distance_apcl, tau_m_apcl_cf, 'o', color=cols_apcl[0], label='apical')
    ax[5].legend(loc='upper right')
    ax[5].set_ylabel('membrane time constant')
    ax[5].set_xlabel('distance to soma ($\mu m$)')

    for a in ax:
        a.set_xlim([0, 600])
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    return fig


def plot_ca_spike4x2():

    cm_neurons = get_colormap()[2]

    l2pyr= get_l2pyr()

    stim = h.IClamp(l2pyr._nrn_sections['apical_tuft'](0.1))
    stim.delay = 100
    stim.dur = 1000
    stim.amp = .5
    t = h.Vector().record(h._ref_t)
    soma_voltg = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)
    clamp_current = h.Vector().record(stim._ref_i)

    dend_voltg = h.Vector().record(l2pyr._nrn_sections['apical_tuft'](0.1)._ref_v)

    soma_ca = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_cai)
            
    dend_ca = h.Vector().record(l2pyr._nrn_sections['apical_tuft'](0.1)._ref_cai)

    h.celsius = 37
    h.finitialize()
    h.continuerun(1000 * ms)

    t_np = t.as_numpy()
    v_np = soma_voltg.as_numpy()
    d_np = dend_voltg.as_numpy()
    dend_peaks = find_peaks(d_np, height=-10, width=2, prominence=10)
    soma_peaks = find_peaks(v_np, height=0, width=0.1)

    fig, ax = plt.subplots(4,2, figsize=(12,10), width_ratios=[1,0.3], constrained_layout=True)

    ax[0,0].plot(t, soma_voltg, color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[0,0].plot(t, dend_voltg, color=cm_neurons[1], linewidth=2)
    ax[0,0].plot(t, clamp_current*10-90, color='r', linewidth=1)        
    ax[0,0].set_xlim([0, 1000])
    ax[0,0].set_ylim([-90, 50])

    ax[0,1].plot(t, soma_voltg, label='soma', color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[0,1].plot(t, dend_voltg, label='apical_tuft bottom', color=cm_neurons[1], linewidth=2)
    ax[0,1].set_xlim([300, 350])
    ax[0,1].legend()
    ax[0,1].set_ylim([-80, 50])

    ax[1,0].plot(t, soma_ca, color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[1,0].plot(t, dend_ca, color=cm_neurons[1], linewidth=2)
    ax[1,0].set_xlim([0, 1000])
    ax[1,0].set_ylim([0, 0.0007])

    ax[1,1].plot(t, soma_ca, color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[1,1].plot(t, dend_ca, color=cm_neurons[1], linewidth=2)
    ax[1,1].set_xlim([300, 350])
    ax[1,1].set_ylim([0, 0.0007])

    del l2pyr
    h('forall delete_section()')
    l2pyr= get_l2pyr()

    stim = h.IClamp(l2pyr._nrn_sections['soma'](0.5))
    stim.delay = 100
    stim.dur = 1000
    stim.amp = .5
    t = h.Vector().record(h._ref_t)
    soma_voltg = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)
    clamp_current = h.Vector().record(stim._ref_i)

    dend_voltg = h.Vector().record(l2pyr._nrn_sections['apical_tuft'](0.1)._ref_v)

    soma_ca = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_cai)
            
    dend_ca = h.Vector().record(l2pyr._nrn_sections['apical_tuft'](0.1)._ref_cai)

    h.celsius = 37
    h.finitialize()
    h.continuerun(1000 * ms)

    t_np = t.as_numpy()
    v_np = soma_voltg.as_numpy()
    d_np = dend_voltg.as_numpy()
    dend_peaks = find_peaks(d_np, width=0.001)
    soma_peaks = find_peaks(v_np, height=0, width=0.1)

    ax[2,0].plot(t, soma_voltg, color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[2,0].plot(t, dend_voltg, color=cm_neurons[1], linewidth=2)
    ax[2,0].plot(t, clamp_current*10-90, color='r', linewidth=1)        
    ax[2,0].set_xlim([0, 1000])
    ax[2,0].set_ylim([-90, 50])

    ax[2,1].plot(t, soma_voltg, label='soma', color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[2,1].plot(t, dend_voltg, label='apical_tuft bottom', color=cm_neurons[1], linewidth=2)
    ax[2,1].set_xlim([300, 350])
    ax[2,1].legend()
    ax[2,1].set_ylim([-80, 50])

    ax[3,0].plot(t, soma_ca, color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[3,0].plot(t, dend_ca, color=cm_neurons[1], linewidth=2)
    ax[3,0].set_xlim([0, 1000])
    ax[3,0].set_ylim([0, 0.0007])

    ax[3,1].plot(t, soma_ca, color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[3,1].plot(t, dend_ca, color=cm_neurons[1], linewidth=2)
    ax[3,1].set_xlim([300, 350])
    ax[3,1].set_ylim([0, 0.0007])

    
    for i in [0, 2]:
        ax[i,0].spines['bottom'].set_bounds([0, 200])
        ax[i,0].spines['bottom'].set_linewidth(1.25)
        ax[i,0].spines['left'].set_bounds([-90, -40])
        ax[i,0].spines['left'].set_linewidth(1.25)
        ax[i,0].set_yticks([])
        ax[i,0].set_xticks([])
        ax[i,1].spines['bottom'].set_bounds([300, 310])
        ax[i,1].spines['bottom'].set_linewidth(1.25)
        ax[i,1].spines['left'].set_bounds([-90, -40])
        ax[i,1].spines['left'].set_linewidth(1.25)
        ax[i,1].set_yticks([])
        ax[i,1].set_xticks([])
        ax[i,0].add_patch(matplotlib.patches.Rectangle((305, -75), 40, 115, linewidth=1.5, edgecolor='k', facecolor='none', linestyle='-.'))
        ax[i,1].add_patch(matplotlib.patches.Rectangle((305, -75), 40, 115, linewidth=1.5, edgecolor='k', facecolor='none', linestyle='-.'))
        

    for i in [1, 3]:
        ax[i,0].spines['bottom'].set_bounds([0, 200])
        ax[i,0].spines['bottom'].set_linewidth(1.25)
        ax[i,0].spines['left'].set_bounds([0, 0.0002])
        ax[i,0].spines['left'].set_linewidth(1.25)
        ax[i,0].set_yticks([])
        ax[i,0].set_xticks([])
        ax[i,1].spines['bottom'].set_bounds([300, 310])
        ax[i,1].spines['bottom'].set_linewidth(1.25)
        ax[i,1].spines['left'].set_bounds([0, 0.0002])
        ax[i,1].spines['left'].set_linewidth(1.25)
        ax[i,1].set_yticks([])
        ax[i,1].set_xticks([])
        ax[i,0].add_patch(matplotlib.patches.Rectangle((305, 0.00005), 40, 0.000635, linewidth=1.5, edgecolor='k', facecolor='none', linestyle='-.'))
        ax[i,1].add_patch(matplotlib.patches.Rectangle((305, 0.00005), 40, 0.000635, linewidth=1.5, edgecolor='k', facecolor='none', linestyle='-.'))
    
    ax = ax.flatten()
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        

    return fig


def plot_ca_spike():

    cm_neurons = get_colormap()[2]

    l2pyr= get_l2pyr()

    stim = h.IClamp(l2pyr._nrn_sections['apical_tuft'](0.1))
    stim.delay = 100
    stim.dur = 1000
    stim.amp = .5
    t = h.Vector().record(h._ref_t)
    soma_voltg = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)
    clamp_current = h.Vector().record(stim._ref_i)

    dend_voltg = h.Vector().record(l2pyr._nrn_sections['apical_tuft'](0.1)._ref_v)

    soma_ca = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_cai)
            
    dend_ca = h.Vector().record(l2pyr._nrn_sections['apical_tuft'](0.1)._ref_cai)

    h.celsius = 37
    h.finitialize()
    h.continuerun(1000 * ms)

    t_np = t.as_numpy()
    v_np = soma_voltg.as_numpy()
    d_np = dend_voltg.as_numpy()
    dend_peaks = find_peaks(d_np, height=-10, width=2, prominence=10)
    soma_peaks = find_peaks(v_np, height=0, width=0.1)

    fig = plt.figure(figsize=(20,5))

    from matplotlib.gridspec import GridSpec
    import matplotlib.gridspec as gridspec

    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 0.5, 0.75])
    gs_small = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0:2,3])

    ax = [None]*8
    ax[0] = fig.add_subplot(gs[0, 0])
    ax[0].plot(t, soma_voltg, color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[0].plot(t, dend_voltg, color=cm_neurons[1], linewidth=2)
    ax[0].plot(t, clamp_current*10-90, color='r', linewidth=1)
    ax[0].set_xlim([0, 1000])
    ax[0].set_ylim([-90, 50])

    #ax[0].set_xlim([100, 400])#ax[0].set_xlim([100, 400])

    ax[1] = fig.add_subplot(gs[0, 1])
    ax[1].plot(t, soma_ca, color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[1].plot(t, dend_ca, color=cm_neurons[1], linewidth=2)
    ax[1].set_xlim([0, 1000])
    ax[1].set_ylim([0, 0.0007])
    ax[2] = fig.add_subplot(gs[0, 2])
    ax[2].plot(t, soma_voltg, label='soma', color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[2].plot(t, dend_voltg, label='apical_tuft bottom', color=cm_neurons[1], linewidth=2)
    ax[2].set_xlim([300, 350])
    ax[2].legend()
    ax[2].set_ylim([-80, 50])

    delay = (dend_peaks[0] - soma_peaks[0])*h.dt
    widths = dend_peaks[1]['widths'] *h.dt

    ax[3] = fig.add_subplot(gs_small[0, 0])
    ax[3].plot(np.ones(len(delay)), delay, 'o', markeredgecolor=cm_neurons[1], label='delay bAP (ms)', color=cm_neurons[1], markersize=9)
    ax[3].set_ylabel('delay (dendrite - soma) (ms)')
    ax[3].set_xlim([0.75, 1.25])
    ax[4] = fig.add_subplot(gs_small[0, 1])
    ax[4].set_xlim([0.75, 1.25])
    ax[4].plot(np.ones(len(widths)), widths, 'o', markeredgecolor=cm_neurons[1],label='width bAP (ms)', color=cm_neurons[1], markersize=9)
    ax[4].set_ylabel('width dendrite spike (ms)')

    # plot soma stimulation
    del l2pyr
    h('forall delete_section()')
    l2pyr= get_l2pyr()

    stim = h.IClamp(l2pyr._nrn_sections['soma'](0.5))
    stim.delay = 100
    stim.dur = 1000
    stim.amp = .5
    t = h.Vector().record(h._ref_t)
    soma_voltg = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)
    clamp_current = h.Vector().record(stim._ref_i)

    dend_voltg = h.Vector().record(l2pyr._nrn_sections['apical_tuft'](0.1)._ref_v)

    soma_ca = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_cai)
            
    dend_ca = h.Vector().record(l2pyr._nrn_sections['apical_tuft'](0.1)._ref_cai)

    h.celsius = 37
    h.finitialize()
    h.continuerun(1000 * ms)

    t_np = t.as_numpy()
    v_np = soma_voltg.as_numpy()
    d_np = dend_voltg.as_numpy()
    dend_peaks = find_peaks(d_np, width=0.001)
    soma_peaks = find_peaks(v_np, height=0, width=0.1)

    ax[5] = fig.add_subplot(gs[1, 0])
    ax[5].plot(t, soma_voltg, color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[5].plot(t, dend_voltg, color=cm_neurons[1], linewidth=2)
    ax[5].plot(t, clamp_current*10-90, color='k', linewidth=1)
    ax[5].set_xlim([0, 1000])
    ax[5].set_ylim([-90, 50])

    ax[6] = fig.add_subplot(gs[1, 1])
    ax[6].plot(t, soma_ca, color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[6].plot(t, dend_ca, color=cm_neurons[1], linewidth=2)
    ax[6].set_xlim([0, 1000])
    ax[6].set_ylim([0, 0.0007])

    ax[7] = fig.add_subplot(gs[1, 2])
    ax[7].plot(t, soma_voltg, label='soma', color=cm_neurons[1], alpha=0.5, linewidth=2)
    ax[7].plot(t, dend_voltg, label='apical_tuft bottom', color=cm_neurons[1], linewidth=2)
    ax[7].set_xlim([350, 400])
    ax[7].set_ylim([-80, 50])

    delay = (dend_peaks[0][1:] - soma_peaks[0])*h.dt
    widths = dend_peaks[1]['widths'][1:] *h.dt

    ax[3].plot(np.ones(len(delay)), delay, 'o', label='delay bAP (ms)', color='w', markeredgecolor=cm_neurons[1], markersize=9)
    ax[3].set_ylabel('delay (dendrite - soma) (ms)')
    ax[3].set_xlim([0.75, 1.25])
    ax[4].set_xlim([0.75, 1.25])
    ax[4].plot(np.ones(len(widths)), widths, 'o', label='width bAP (ms)', color='w', markeredgecolor=cm_neurons[1], markersize=9)
    ax[4].set_ylabel('width dendrite spike (ms)')
    ax[3].set_ylim([-0.6, 1.8])
    ax[3].set_yticks(np.arange(-0.6, 1.81, 0.6))
    ax[3].set_xticks([1])
    ax[3].set_xticklabels([])
    ax[4].set_xticks([1])
    ax[4].set_xticklabels([])
    ax[4].set_ylim([3, 5])
    ax[4].set_yticks(np.arange(3, 5.1, 1))
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    for i in [0, 5]:
        ax[i].spines['bottom'].set_bounds([0, 200])
        ax[i].spines['bottom'].set_linewidth(1.25)
        ax[i].spines['left'].set_bounds([-90, -40])
        ax[i].spines['left'].set_linewidth(1.25)
        ax[i].set_yticks([])
        ax[i].set_xticks([])

    for i in [1, 6]:
        ax[i].spines['bottom'].set_bounds([0, 200])
        ax[i].spines['bottom'].set_linewidth(1.25)
        ax[i].spines['left'].set_bounds([0, 0.0002])
        ax[i].spines['left'].set_linewidth(1.25)
        ax[i].set_yticks([])
        ax[i].set_xticks([])

    for i in [2, 7]:
        ax[i].spines['bottom'].set_bounds([300, 310])
        ax[i].spines['bottom'].set_linewidth(1.25)
        ax[i].spines['left'].set_bounds([-80, -30])
        ax[i].spines['left'].set_linewidth(1.25)
        ax[i].set_yticks([])
        ax[i].set_xticks([])

    return fig