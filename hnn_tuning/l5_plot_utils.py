"""" Plotting functions for exploring Hay model and HNN """
from neuron import h
from neuron.units import ms, mV
from scipy.signal import find_peaks

import os.path as op
from ._paths import HNN_CORE_MOD, results_dir

pth_mod = str(HNN_CORE_MOD)
figpth = str(results_dir('single_cell_sim', 'l5_ET'))

import matplotlib.pyplot as plt
import numpy as np

from functools import partial

from .cell_builder_utils import l5_params, Cell, _get_dends, _get_pyr_soma
from .l5_mechs_utils import Mechs, Ih_condu_hay, linear_gbar

from matplotlib import colormaps

def get_colormap_l5(cmap_name='inferno'):
    """ Get colormap for plotting """

    cmap = colormaps['inferno']
    cols_dist = cmap.colors[0:len(cmap.colors):6]
    col_basal = np.array(([56,108,176], [102,166,30], [231,41,138], [231,41,138]))/255
    cols_experiments = np.array(([27, 158, 119], [217,95,2], [117,112,179], [231,41,138], [102,166,30]))/255

    l5pyr = get_l5pyr()
    distance = []
    h.distance(sec = l5pyr._nrn_sections['soma'])
    section_names = ['soma', 'apical_trunk','apical_1', 'apical_2', 'apical_tuft']

    for section in section_names:
        for seg in l5pyr._nrn_sections[section]:
            distance.append(h.distance(seg))

    im = plt.pcolormesh(np.tile(np.array(distance),(2,1)), cmap=cmap_name, vmin=0, vmax=2000)
    cb = plt.colorbar(im, orientation='horizontal', pad=0.2)
    cb.set_ticks([0, 1000, 2000])
    plt.savefig(op.join(figpth, 'l5_distance_colormap.pdf'), format='pdf', dpi=300)
    plt.clf()

    return im, cols_dist, col_basal, cols_experiments

def get_model_params():

    return {'soma': {'NaTs2_t_hay2011': 20_400e-4/2*1.5,
                                'SKv3_1_hay2011': 6_930e-4/2,
                                'Nap_Et2_hay2011': 17.2e-4/2*0.11,
                                'Ca_HVA_hay2011': 9.92e-04/2,
                                'Ca_LVAst_hay2011': 34.3e-4/2,
                                'SK_E2_hay2011': 441e-4*15,                      # increase this one to get burst followed by regular firing
                                'Ih_hay2011': 1e-4/2,
                                'Im_hay2011': 1e-4,
                                'K_Pst_hay2011': 22.3e-4/2,#*0.1,
                                'K_Tst_hay2011': 812e-4/2,#*0.1,z
                                'pas': {'g': .338e-4/2/2,
                                          'e': -90}},
                        'basal': {'NaTs2_t_hay2011': partial(linear_gbar,0.0102,0,255),
                                'SKv3_1_hay2011': partial(linear_gbar,0,0.0001305,255),
                                  'Ih_hay2011': 5.14e-5/4*2,
                                  'pas': {'g': 1.75e-5/2*3,
                                          'e': -90}},
                        'dend': {'NaTa_t_hay2011': 0.0213/2,
                                'SKv3_1_hay2011': 0.000261/2*2,
                                'Ca_HVA_hay2011': partial(linear_gbar, 2.78e-5/2., 2.78e-5/2*12.0, 1500, hotzone=[1500, 1700], hotzone_factor=4.5),
                                'Ca_LVAst_hay2011': partial(linear_gbar, 93.5e-6/2, 93.5e-6/2*2.25, 1500, hotzone=[1500, 1700], hotzone_factor=2.25),
                                'SK_E2_hay2011': 0.0012/2*6,#*2,#*1.6,
                                'Ih_hay2011': partial(Ih_condu_hay, 1e-4/2*1.2),   
                                'Im_hay2011': 0.0000675/2,
                                'K_Pst_hay2011': 0,
                                'K_Tst_hay2011': 0,
                                'pas': {'g': 0.0000589/1.5,
                                          'e': -85}}}

def get_l5pyr(v_init={
                "apical_trunk": -71.48534399721858,
                "apical_1": -70.35766714762656,
                "apical_2": -66.1410056008678,
                "apical_tuft": -61.46830840811679,
                "basal_1": -71.63003121190522,
                "basal_2": -71.75791098784391,
                "basal_3": -71.75791098784391,
                "apical_oblique": -71.51514840992614,
                "soma": -71.54401603093514,
            }, add_syn=None, syn_loc=None):
    # get paramaters
    section_params, end_pts, cell_tree = l5_params()

    # build sections
    section_names = list(end_pts.keys())

    sections = _get_dends(section_params, 'L5Pyr', section_names, end_pts, v_init=v_init)
    sections['soma'] = _get_pyr_soma(section_params, 'L5Pyr', end_pts['soma'], v_init=v_init['soma'])

    mech_class = Mechs(pth_mod)

    l5pyr= Cell(sections, cell_tree, cell_type='L5_pyramidal', gid=0)
    l5pyr.build(add_syn=add_syn, syn_loc=syn_loc)

    model_params = get_model_params()
    l5pyr = mech_class.add_mechs(l5pyr, model_params)

    return l5pyr

def fig_conductances(col_basal, cell_type='ET'):

    model = get_l5pyr()
    apical_sections=['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft']
    distance_apical = []
    gbar_Ih_apical = []
    gbar_Ca_HVA_apical = []
    gbar_Ca_LVA_apical = []

    h.distance(sec = model._nrn_sections['soma'])
    for section in apical_sections:
        for seg in model._nrn_sections[section]:
            distance_apical.append(h.distance(seg.x, sec=seg.sec))
            gbar_Ih_apical.append(seg.gbar_Ih_hay2011)
            gbar_Ca_HVA_apical.append(seg.gbar_Ca_HVA_hay2011)
            gbar_Ca_LVA_apical.append(seg.gbar_Ca_LVAst_hay2011)

    fig, ax = plt.subplots(2,3,figsize=(11,5))
    ax[0,0].plot(distance_apical, gbar_Ih_apical, color='k')
    ax[0,0].set_ylim([0.0,0.05])
    ax[0,0].set_yticks([0.0,0.025,0.05])
    ax[0,0].set_ylabel('$gbar_{Ih} (S/cm^2)$')
    ax[0,0].set_xlabel('distance to soma ($\mu m$)')

    ax[0,1].plot(distance_apical, gbar_Ca_HVA_apical, color='k')
    ax[0,1].set_ylabel('$gbar_{Ca_{HVA}} (S/cm^2)$')
    ax[0,1].set_xlabel('distance to soma ($\mu m$)')
    ax[0,1].set_ylim([0.0,0.0008])
    ax[0,1].set_yticks([0.0,0.0008/2,0.0008])

    ax[0,2].plot(distance_apical, gbar_Ca_LVA_apical, color='k')
    ax[0,2].set_ylabel('$gbar_{Ca_{LVA}} (S/cm^2)$')
    ax[0,2].set_xlabel('distance to soma ($\mu m$)')
    ax[0,2].set_ylim([0.0,0.0003])
    ax[0,2].set_yticks([0.0,0.0003/2,0.0003])

    for a in ax[0]:
        a.set_xlim([0, 2000])
        a.set_xticks([0, 1000, 2000])


    l_stim = 150

    section_names = ['apical_oblique', 'basal_1','basal_2', 'basal_3']

    voltg = [None]*len(section_names)

    stim = h.IClamp(model._nrn_sections['soma'](0.5))
    stim.delay = 20
    stim.dur = 5
    stim.amp = 3

    v_soma = h.Vector().record(model._nrn_sections['soma'](0.5)._ref_v)
    

    h.distance(sec = model._nrn_sections['soma'])
    for s, section in enumerate(section_names):
        voltg[s] = dict()
        for seg in model._nrn_sections[section]:
            voltg[s][seg.x] = h.Vector().record(seg._ref_v)

    t = h.Vector().record(h._ref_t)           
    h.celsius = 37
    h.finitialize()
    h.continuerun(l_stim * ms)

    v_soma = v_soma.as_numpy()

    ax12_tw = ax[1,2].twinx()
    ax12_tw.set_ylabel('% of soma AP')
    for s, section in enumerate(section_names):
        Nagbar = []
        Kbar = []   
        amp = []
        amp_percent = []
        dist = []
        for seg in model._nrn_sections[section]:
            dist.append(h.distance(seg.x, sec=seg.sec))

            if cell_type == 'ET':
                Nagbar.append(seg.gbar_NaTs2_t_hay2011)
            elif cell_type == 'IT':
                Nagbar.append(seg.gbar_NaTs2_t_32d_hay2011)
            Kbar.append(seg.gbar_SKv3_1_hay2011)
            

        # get amplitude
        for segx in voltg[s].keys():
            vmax = np.max(voltg[s][segx].as_numpy())
            vmin = np.min(voltg[s][segx].as_numpy())
            amp.append(vmax-vmin)
            amp_percent.append((vmax - vmin)/(np.max(v_soma)-np.min(v_soma)) * 100)

        ax[1,0].plot(dist, Nagbar, 'o', color=col_basal[s], alpha=0.5)
        ax[1,0].set_xlabel('distance from soma')
        ax[1,0].set_ylabel('$gbar_{Na} (S/cm^2)$')
        ax[1,0].set_ylim([-0.0005, 0.009])
        ax[1,0].set_yticks([0, 0.0045, 0.009])

        ax[1,1].plot(dist, Kbar, 'o', color=col_basal[s], alpha=0.5)
        ax[1,1].set_xlabel('distance from soma')
        ax[1,1].set_ylabel('$gbar_{K+} (S/cm^2)$')
        ax[1,1].set_ylim([0, 0.0002])
        ax[1,1].set_yticks([0, 0.0001, 0.0002])
        ax[1,2].plot(dist, amp, 'o', color=col_basal[s], alpha=0.5, label=section)
        ax[1,2].set_xlabel('distance from soma')
        ax[1,2].set_ylabel('BAP amplitude (mV)')
        ax[1,2].set_ylim([70, 90])
        ax[1,2].set_yticks([70, 80, 90])
        ax12_tw.plot(dist, amp_percent, 'o', color=col_basal[s], alpha=0.01)
        ax12_tw.set_ylim([60, 80])
        ax12_tw.set_yticks([60, 70, 80])

        ax[1,2].legend()

        for a in ax[1]:
            a.set_xlim([0, 400])
            a.set_xticks([0, 200, 400])

        
    ax = ax.ravel()
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    
    fig.tight_layout()

    return fig

def fig_caspike_critical_frequency(cols):

    fig, ax = plt.subplots(3, 2, figsize=(6,4))
    h('forall delete_section()')
    nums_pulse = [2, 4, 4]
    time_diff = [10, 10, 13]
    l_stim = 500

    section_names = ['apical_tuft', 'soma']

    voltg = [None]*len(section_names)
    ca = [None]*len(section_names)

    for n, num_pulse in enumerate(nums_pulse):
        
        l5pyr = get_l5pyr()

        stim = [None]*num_pulse
        t = h.Vector().record(h._ref_t)
        for i in range(num_pulse):
            stim[i] = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
            stim[i].delay = 20+(time_diff[n]*i)
            stim[i].dur = 5
            stim[i].amp = 3

        voltg[0] = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)
        ca[0] = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_cai)

        voltg[1] = h.Vector().record(l5pyr._nrn_sections['apical_tuft'](0.1)._ref_v)
        ca[1] = h.Vector().record(l5pyr._nrn_sections['apical_tuft'](0.1)._ref_cai)

        h.celsius = 37
        h.finitialize()
        h.continuerun(l_stim * ms)

        clamp_current = np.zeros_like(t.as_numpy())
        for i in range(num_pulse):
            clamp_current[(t.as_numpy() > stim[i].delay) & (t.as_numpy() < stim[i].delay+stim[i].dur)] = stim[i].amp

        
        ax[n, 0].plot(t.as_numpy(), np.zeros_like(t.as_numpy()), color='black', alpha=0.5, linewidth=0.5, linestyle='--')
        ax[n, 0].plot(t, voltg[0], color='k', linewidth=1)
        ax[n, 0].plot(t.as_numpy(), clamp_current*2-80, color='black', alpha=0.75, linewidth=0.75)
        ax[n, 1].plot(t, ca[0], color='k', linewidth=1)

        v_max = np.max(voltg[1].as_numpy())
        t_max = np.argmax(voltg[1].as_numpy())
        ax[n, 0].text(t[t_max], v_max, f'{v_max:.2f}', fontsize=8)
        ax[n, 0].plot(t, voltg[1], color='#5c72b7', alpha=1, linewidth=1)
        ax[n, 1].plot(t, ca[1], label=f'{num_pulse} pulses, {time_diff[n]} ms apart', color='#5c72b7', alpha=1, linewidth=1)
        
        ax[n, 0].set_xlim([15, 90])
        ax[n, 1].set_yticks(np.arange(0, 0.0005, 0.0001))
        ax[n, 1].set_xlim([15, 90])
        ax[n, 0].set_yticks(np.arange(-70, 70, 20))

        for i in range(2):
            ax[n, i].spines['top'].set_visible(False)
            ax[n, i].spines['right'].set_visible(False)
            ax[n, i].spines['bottom'].set_bounds(20, 30)
            ax[n, i].set_yticks([])
            ax[n, i].set_xticks([])
            ax[n, i].spines['bottom'].set_linewidth(1.25)
            ax[n, i].spines['left'].set_linewidth(1.25)
        
        ax[n, 0].spines['left'].set_bounds(-85, -35)
        ax[n, 1].spines['left'].set_bounds(0, 0.0001)


    fig.tight_layout()

    return fig

def fig_ca_spike_measures():

    fig, ax = plt.subplots(1,2, figsize=(8,2), constrained_layout=True)

    # depolarization integral
    time_diff = np.linspace(17, 7, 10)
    freqs = np.zeros_like(time_diff)
    apcl_integr = np.zeros_like(time_diff)
    for n, td in enumerate(time_diff):
        
        l5pyr = get_l5pyr()

        t = h.Vector().record(h._ref_t)
        num_pulse = 4
        stim = [None]*4
        t = h.Vector().record(h._ref_t)
        for i in range(num_pulse):
            stim[i] = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
            stim[i].delay = 20+(td*i)  
            stim[i].dur = 5
            stim[i].amp = 3

        v_apcl = h.Vector().record(l5pyr._nrn_sections['apical_tuft'](0.5)._ref_v)
        h.celsius = 37
        h.finitialize()
        h.continuerun(1020 * ms)

        freqs[n] = 1000/td
        v_apcl = v_apcl.as_numpy()
        pk = find_peaks(v_apcl, prominence=15)

        apcl_integr[n] = np.sum(v_apcl-v_apcl[pk[1]['left_bases'][0]])*h.dt*0.001
    ax[0].plot(freqs, apcl_integr, 'o', color='black', markersize=7)
    ax[0].set_xlim([np.min(freqs)-1, np.max(freqs)+1])
    ax[0].set_ylim([0.4, 1.6])
    ax[0].set_yticks([0.4, 1.0, 1.6])
    ax[0].set_ylabel('membr. pot. integral (mV$\cdot$s)')
    ax[0].set_xlabel('input frequency (Hz)')

    # input threshold
    section_names = ['soma', 'apical_trunk', 'apical_1', 'apical_2', 'apical_tuft']
    l5pyr = get_l5pyr()
    threshold = []
    for s, section in enumerate(section_names):
        for i, seg in enumerate(l5pyr._nrn_sections[section]):
            tuft_mp = -70 # initialize to a value below threshold
            amp = 0.1 # starting value for stimulation amplitude
            
            while tuft_mp < 0: # has MP reached critical threshold?
                del l5pyr
                l5pyr = get_l5pyr()

                stim = h.IClamp(l5pyr._nrn_sections[section](seg.x))
                stim.delay = 20 
                stim.dur = 100
                stim.amp = amp

                v = h.Vector().record(l5pyr._nrn_sections[section](seg.x)._ref_v)
                t = h.Vector().record(h._ref_t)

                h.celsius = 37
                h.finitialize()
                h.continuerun(100 * ms)

                tuft_mp =np.max(v.as_numpy())
                amp += 0.2

            threshold.append(amp)

    # get distance from soma (this does not work in any of the above loops because I keep deleting l5pyr)
    distance = []
    h.distance(sec = l5pyr._nrn_sections['soma'])
    for section in section_names:
        for i, seg in enumerate(l5pyr._nrn_sections[section]):
            distance.append(h.distance(seg.x, sec=seg.sec)) 

    ax[1].plot(distance, threshold, 'o', color='k',markersize=7)
    ax[1].set_ylabel('threshold (nA)')
    ax[1].set_xlabel('distance from soma ($\mu m$)')

    return fig

def fig_caspike_coincidence(cols):

    fig, ax = plt.subplots(4, 2, figsize=(6, 6))

    nums_pulse = [1, 1, 2, 1]
    pulse_sect = [['soma'], ['apical_tuft'],['soma', 'apical_tuft'], ['apical_tuft']]
    ampl = [[3], [0.5], [3, 0.5], [2], ]
    time_diff = [[0], [0], [0, 5], [0]]
    l_stim = 150

    section_names = ['apical_tuft', 'soma']

    voltg = [None]*len(section_names)
    ca = [None]*len(section_names)

    for n, num_pulse in enumerate(nums_pulse):
        
        l5pyr = get_l5pyr()

        stim = [None]*num_pulse
        t = h.Vector().record(h._ref_t)
        
        for i in range(num_pulse):
            stim[i] = h.IClamp(l5pyr._nrn_sections[pulse_sect[n][i]](0.5))
            stim[i].delay = 20+(time_diff[n][i])
            stim[i].dur = 5
            stim[i].amp = ampl[n][i]


        voltg[0] = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)
        ca[0] = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_cai)
        voltg[1] = h.Vector().record(l5pyr._nrn_sections['apical_tuft'](0.5)._ref_v)
        ca[1] = h.Vector().record(l5pyr._nrn_sections['apical_tuft'](0.5)._ref_cai)

        h.celsius = 37
        h.finitialize()
        h.continuerun(l_stim * ms)

        clamp_current = np.zeros((num_pulse, len(t.as_numpy())))
        for i in range(num_pulse):
            clamp_current[i][(t.as_numpy() > stim[i].delay) & (t.as_numpy() < stim[i].delay+stim[i].dur)] = stim[i].amp*3

        for i in range(num_pulse):
            if pulse_sect[n][i] == 'soma':
                col = 'black'
                c = clamp_current[i]*2-95
            else:
                col = 'red'
                c = -clamp_current[i]*2-73
            ax[n, 0].plot(t.as_numpy(), c, color=col, alpha=0.75, linewidth=1)

            # if pulse_sect[n][i] == 'soma':
            #     col = 'black'
            #     c = clamp_current[i]*0.00001-0.00015
            # else:
            #     col = 'red'
            #     c = -clamp_current[i]*0.00001-0.00015
            # ax[n, 1].plot(t.as_numpy(), c, color=col, alpha=0.75, linewidth=1)

        ax[n, 0].plot(t.as_numpy(), np.zeros_like(t.as_numpy()), color='black', alpha=0.5, linewidth=0.5, linestyle='--')
        ax[n, 0].plot(t, voltg[0], color='k', linewidth=1)

        v_max = np.max(voltg[1].as_numpy())
        t_max = np.argmax(voltg[1].as_numpy())
        ax[n, 0].text(t[t_max], v_max, f'{v_max:.2f}', fontsize=8)
        ax[n, 0].plot(t, voltg[1], color='#5c72b7', alpha=1, linewidth=1)

        ax[n, 1].plot(t, ca[0], color='k', linewidth=1)
        ax[n, 1].plot(t, ca[1], color='#5c72b7', alpha=1, linewidth=1)
                                
        ax[n, 0].set_xlim([15, 90])
        ax[n, 1].set_yticks(np.arange(0, 0.0005, 0.0001))
        ax[n, 1].set_xlim([15, 90])
        ax[n, 0].set_yticks(np.arange(-70, 70, 20))
        ax[n, 0].set_ylim(-95, 70)

        for i in range(2):
            ax[n, i].spines['top'].set_visible(False)
            ax[n, i].spines['right'].set_visible(False)
            ax[n, i].spines['bottom'].set_bounds(20, 30)
            ax[n, i].set_yticks([])
            ax[n, i].set_xticks([])
            ax[n, i].spines['bottom'].set_linewidth(1.25)
            ax[n, i].spines['left'].set_linewidth(1.25)
        
        ax[n, 0].spines['left'].set_bounds(-85, -35)
        ax[n, 1].spines['left'].set_bounds(0, 0.0001)


    fig.tight_layout()

    return fig

def fig_electro_measures():

    l5pyr = get_l5pyr()
    
    # Voltage sag as a function of distance (Beaulieu-Laroche et al. 2018)
    amp = -.5
    sag_ratio = []

    section_names = ['soma', 'apical_trunk', 'apical_1', 'apical_2', 'apical_tuft']
    for s, section in enumerate(section_names):

        for i, seg in enumerate(l5pyr._nrn_sections[section]):
            l5pyr = get_l5pyr()

            stim = h.IClamp(l5pyr._nrn_sections[section](seg.x))
            stim.delay = 20
            stim.dur = 150
            stim.amp = amp

            t = h.Vector().record(h._ref_t)
            soma_voltg = h.Vector().record(l5pyr._nrn_sections[section](0.5)._ref_v)

            h.celsius = 37
            h.finitialize()
            h.continuerun(250 * ms)

            steady_idx = np.argmin(np.abs(t-150))
            soma_v = soma_voltg.as_numpy()

            pk = np.min(soma_v)
            
            steady_state = soma_v[steady_idx]

            # Sag ratio - see Beaulieu-Laroche, 2018
            sag_ratio.append((pk-steady_state)/pk)
    
    # input resistance as a function of distance (Beaulieu-Laroche et al. 2018)
    inp_res = []
    
    for s, section in enumerate(section_names):
        for i, seg in enumerate(l5pyr._nrn_sections[section]):
            tuft_mp = -70 # initialize to a value below threshold
            amp = 0.1 # starting value for stimulation amplitude
            del l5pyr 
            l5pyr = get_l5pyr()

            impedance = h.Impedance()
            h.celsius = 37
            h.finitialize()
            impedance.loc(seg.x, sec=l5pyr._nrn_sections[section])
            impedance.compute(0)
            inp_res.append(impedance.input(l5pyr._nrn_sections[section](seg.x)))
    
    # get distance from soma (this does not work in any of the above loops because I keep deleting l5pyr)
    distance = []
    h.distance(sec = l5pyr._nrn_sections['soma'])
    for section in section_names:
        for i, seg in enumerate(l5pyr._nrn_sections[section]):
            distance.append(h.distance(seg.x, sec=seg.sec)) 

    fig, ax = plt.subplots(1,2, figsize=(8, 3), constrained_layout=True)

    ax[0].plot(distance, sag_ratio, 'o', color='k',markersize=7)
    ax[0].set_ylabel('sag ratio')
    ax[0].set_xlabel('distance from soma ($\mu m$)')
    ax[1].plot(distance, inp_res, 'o', color='k',markersize=7)
    ax[1].set_xlabel('distance from soma ($\mu m$)')
    ax[1].set_ylabel('input resistance (M$\Omega$)')

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_xlim([0, 2000])
        a.set_xticks([0, 1000, 2000])
    
    return fig, distance

def fig_compartments(model, clamp_current, l_stim=200):

    section_names = ['apical_tuft', 'apical_2', 'apical_1', 'apical_trunk', 'soma']

    voltg = [None]*len(section_names)
    ca = [None]*len(section_names)
    for i, section in enumerate(section_names):
        voltg[i] = h.Vector().record(model._nrn_sections[section](0.5)._ref_v)
        ca[i] = h.Vector().record(model._nrn_sections[section](0.5)._ref_cai)

    t = h.Vector().record(h._ref_t)


    h.celsius = 37
    h.finitialize()
    h.continuerun(l_stim * ms)

    fig, ax = plt.subplots(2,3, figsize=(13,8), height_ratios=[0.25, 1], sharex=True)

    for c in clamp_current:
        ax[0,0].plot(t,c, color='k')
        ax[0,1].plot(t,c, color='k')
        ax[0,2].plot(t,c, color='k')

    ax[0,0].set_ylabel('clamp \n current')
    yticks_v = []
    yticklabels_v = []

    yticks_ca = []
    yticklabels_ca = []

    cv = 0
    cca = 0
    for i,section in enumerate(section_names):
        
        ax[1,0].plot(t,voltg[i] + cv)
        if section == 'apical_tuft':
            #ax[1,0].plot(t,np.ones_like(ca[i]) * cv, color='k', linestyle='-.')
            v_max = np.max(voltg[i].as_numpy())
            t_max = np.argmax(voltg[i].as_numpy())
            ax[1,0].text(t[t_max], v_max, f'{v_max:.2f}', fontsize=8)


        yticks_v.append(-75+cv)
        yticks_v.append(50+cv)

        yticklabels_v.append(str(np.round(-75)))
        yticklabels_v.append(str(np.round(50)))

        cv -= 200

        ax[1,1].plot(t,ca[i] + cca, label=section)
        


        yticks_ca.append(min(ca[i])+cca)
        yticks_ca.append(max(ca[i])+cca)

        yticklabels_ca.append(str(min(ca[i])))
        yticklabels_ca.append(str(np.round(max(ca[i]),5)))

        cca -= 6e-4


    ax[1,0].set_yticks(yticks_v)
    ax[1,0].set_yticklabels(yticklabels_v)
    ax[1,0].set_ylabel('voltage $\mu V$')
    ax[1,0].set_xlabel('time (ms)')


    ax[1,1].set_yticks(yticks_ca)
    ax[1,1].set_yticklabels(yticklabels_ca)
    ax[1,1].set_ylabel('calcium $mM$')
    ax[1,1].set_xlabel('time (ms)')

    

     
    fig.legend(loc='lower right')


    fig.tight_layout()

    return fig

def _hammfilt(x, winsz):
    """Convolve with a hamming window."""
    assert len(x) > winsz
    win = np.hamming(winsz)
    win /= sum(win)
    return np.convolve(x, win, 'same')