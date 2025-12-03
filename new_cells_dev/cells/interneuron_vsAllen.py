"""Interneuron vs Allen data"""

import neuron
from neuron import h
import matplotlib.pyplot as plt
from neuron.units import ms, mV
import numpy as np
pth_mod_inh = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/local_hnn/hnn_core/mod'
pth_data = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/data/interneuron'
figpth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/new_cells_dev/figures/interneuron/'
import os
#os.mkdir(figpth)
import pickle
plt.rcParams['svg.fonttype'] = 'none'

import sys
sys.path.append('/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/utils')
from cell_builder_utils import l5_params, _get_dends, _get_pyr_soma, Cell, _get_basket_soma
from inh_mechs_utils import Mechs as inhmechs
from functools import partial

from hnn_core.viz import plot_cell_morphology

import os
import pandas as pd
from scipy import signal
from l23_plot_utils import get_colormap
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

h.load_file('stdrun.hoc')


plt.rcParams['svg.fonttype'] = 'none'
# Set global font sizes
plt.rcParams['font.size'] = 16          # Default text size
plt.rcParams['axes.titlesize'] = 16      # Title size
plt.rcParams['axes.labelsize'] = 16      # Axis labels (xlabel, ylabel)
plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 16     # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 16     # Legend text


# Build inhibitory cell

section = dict()
section['soma'] = _get_basket_soma()

mech_class = inhmechs(pth_mod_inh)

# Cumberland 2023 and Golomb 2007 parameters
inh_params = {'soma': {'nas': 0.1125, 
                'kdr': 0.225*2,
                'kd': 0.00039*2,
                'Ih': 2e-5*9,
                'pas': {'g': .0001,
                        'e': -70}}}

def exp_func(t, tau, A, C):
    return A * (1 - np.exp(-t / tau)) + C

"""Load Allen data"""
with open(os.path.join(pth_data,'ephys_inh.pkl'), 'rb') as f:
    inh_allen = pickle.load(f)
ephys_measures = dict()


cm_neurons = get_colormap('inferno')[2]
cols = np.array(([27, 158, 119], [217,95,2], [117,112,179], [231,41,138], [102,166,30]))/255


## RMP
fig, ax = plt.subplots(2, 3, width_ratios=[0.5, 0.5, 1], figsize=(16, 6), constrained_layout=True)
ax = ax.flatten()

inh_cell = Cell(section, cell_tree=None, cell_type='Inhibitory', gid=0)
inh_cell.build()
inh_cell = mech_class.add_mechs(inh_cell, inh_params)

t = h.Vector().record(h._ref_t)
soma_v = h.Vector().record(inh_cell._nrn_sections['soma'](0.5)._ref_v)

h.celsius = 37
h.dt = 0.01
h.finitialize(-65 * mV)
h.continuerun(200 * ms)


# RMP
ephys_measures['RMP'] = soma_v[-3]

print('RMP:', ephys_measures['RMP'], flush=True)

ax[0].plot(t, soma_v, color=cm_neurons[2], linewidth=2)
ax[0].plot(t.as_numpy()[-3],soma_v[-3],'o', color='k', markersize=9)
ax[0].set_ylabel('membrane \n potential (mV)')
ax[0].set_xlabel('time (ms)')

## Action potential
inh_cell = Cell(section, cell_tree=None, cell_type='Inhibitory', gid=0)
inh_cell.build()
inh_cell = mech_class.add_mechs(inh_cell, inh_params)

stim = h.IClamp(inh_cell._nrn_sections['soma'](0.5))
stim.delay = 5
stim.dur = 3
stim.amp = 0.1
clamp_current = h.Vector().record(stim._ref_i)

t = h.Vector().record(h._ref_t)
v_soma = h.Vector().record(inh_cell._nrn_sections['soma'](0.5)._ref_v)

h.celsius = 37
h.dt = 0.01
h.finitialize()
h.continuerun(20 * ms)

ap, properties = find_peaks(v_soma.as_numpy(), height=0, width=0)

v_np = v_soma.as_numpy()
t_np = t.as_numpy()
smooth_v = signal.savgol_filter(np.diff(v_np), 201, 3, mode='constant')

ap_thresh_idx = np.where(smooth_v/np.diff(t_np)[0]>20)[0][0]+1

ap_thresh = v_np[ap_thresh_idx]

if len(ap) > 0:
    ap_peak = properties['peak_heights'][0]
else:
    ap_peak = v_soma[-1]

start_ap = int(ap[0])
stop_ap = np.argmin(np.abs(t_np-(t_np[ap[0]]+5)))
elbow_p = int(np.argmin(v_np[start_ap:stop_ap]) + start_ap)

upstroke = np.mean(np.gradient(v_np[ap_thresh_idx:int(ap[0])]))
downstroke = np.mean(np.gradient(v_np[int(ap[0]):elbow_p]))

ap_up_down = np.abs(upstroke/downstroke)


ax[1].plot(t, v_soma, color=cm_neurons[2], linewidth=2)
ax[1].plot(t[np.argmax(v_soma)], ap_peak, 'o', color=cols[0], markersize=9)
ax[1].set_ylim([-85, 50])
ax[1].set_yticks([-70,0, 50])
ax[1].plot(t[ap_thresh_idx], ap_thresh, 'o', color=cols[1], markersize=9)
ax[1].set_xlim([0, 20])
ax[1].set_xticks([0, 10, 20])
ax[1].set_xlabel('time (ms)')

ax[1].plot(t.as_numpy()[elbow_p], v_soma.as_numpy()[elbow_p], 'o', color=cols[2], markersize=9)

ap_width = properties['widths'][0]*h.dt
ephys_measures['ap_width'] = ap_width
ephys_measures['upstroke_downstroke'] = ap_up_down
ephys_measures['ap_threshold'] = v_soma[ap_thresh_idx]

# ISI
amp_isi = np.mean(inh_allen['max_amp_isi'].dropna().values)
inh_cell = Cell(section, cell_tree=None, cell_type='Inhibitory', gid=0)
inh_cell.build()
inh_cell = mech_class.add_mechs(inh_cell, inh_params)
stim = h.IClamp(inh_cell._nrn_sections['soma'](0.5))
stim.delay = 20
stim.dur = 1000
stim.amp = amp_isi

t = h.Vector().record(h._ref_t)
soma_v = h.Vector().record(inh_cell._nrn_sections['soma'](0.5)._ref_v)

h.celsius = 37
h.finitialize()
h.continuerun(1020 * ms)

ax[2].plot(t,soma_v, color=cm_neurons[2], linewidth=2)
ax[2].set_xlim([0, 1000])
ax[2].set_yticks([-70,0, 50])
ax[2].set_ylim([-80, 50])
ax[2].set_xlabel('time (ms)')

aps = find_peaks(soma_v,height=0)[0]
ephys_measures['first_isi'] = np.diff(t.as_numpy()[aps])[0]
ephys_measures['mean_isi'] = np.mean(np.diff(t.as_numpy()[aps]))

## FR slope

l_stim = 1000

max_amp_isi = np.max(inh_allen['max_amp_isi'].dropna().values)

ampl = np.array([amp_isi/3, amp_isi/1.5, amp_isi, max_amp_isi/1.5, max_amp_isi])

soma_v = np.zeros((len(ampl),int(1/h.dt*l_stim+1)))
fr = np.zeros_like(ampl)

for i,amp in enumerate(ampl):

    inh_cell = Cell(section, cell_tree=None, cell_type='Inhibitory', gid=0)
    inh_cell.build()
    inh_cell = mech_class.add_mechs(inh_cell, inh_params)

    stim = h.IClamp(inh_cell._nrn_sections['soma'](0.5))
    stim.delay = 20
    stim.dur = 1000
    stim.amp = amp
    clamp_current = h.Vector().record(stim._ref_i)

    t = h.Vector().record(h._ref_t)
    v_soma = h.Vector().record(inh_cell._nrn_sections['soma'](0.5)._ref_v)

    h.celsius = 37
    h.finitialize()
    h.continuerun((l_stim+stim.delay)* ms)

    pk = find_peaks(v_soma, height=0)

    fr[i] = len(pk[0])


ax[3].plot(ampl,fr, color=cm_neurons[2], linewidth=2)
ax[3].set_ylabel('FR (Hz)')
ax[3].set_xlabel('amplitude (nA)')
ax[3].set_xlim([ampl[0], ampl[-1]])
pA = 0.001 # nA (convert nA to pA)
ephys_measures['FR_slope'] = np.polyfit(ampl,fr,1)[0]
ephys_measures['FR_slope'] *= pA


# Hyperpolarization experiments
sag_stim_amp = np.unique(np.concatenate(inh_allen['sag_stim_amp'].dropna().values))
tau_ms = np.zeros_like(sag_stim_amp)
steady_state = np.zeros_like(sag_stim_amp)
min_membr_pot = np.zeros_like(sag_stim_amp)

for i, amp in enumerate(sag_stim_amp):
    inh_cell = Cell(section, cell_tree=None, cell_type='Inhibitory', gid=0)
    inh_cell.build()
    inh_cell = mech_class.add_mechs(inh_cell, inh_params)

    h.distance(sec = inh_cell._nrn_sections['soma'])

    stim = h.IClamp(inh_cell._nrn_sections['soma'](0.5))
    stim.delay = 50
    stim.dur = 500
    stim.amp = amp

    v = h.Vector().record(inh_cell._nrn_sections['soma'](0.5)._ref_v)

    t = h.Vector().record(h._ref_t)

    h.celsius = 37
    h.finitialize()
    h.continuerun(650 * ms)

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
    popt, pcov = curve_fit(exp_func, t_rise, v_rise, p0=[2, abs(delta_v), v_rise[0]])
    tau_ms[i] = popt[0] # This is the fitted time constant in ms

    ax[5].plot(t_np, v_np, color=cm_neurons[2], linewidth=2)
    idx = np.argmin(np.abs(t_np-tau_ms[i]))
    ax[5].plot(t_np[idx+start_idx], v_np[idx+start_idx], 'o', color='k', markersize=9)

ax[5].set_xlim([0, 650])
ax[5].set_ylabel('membrane \n potential (mV)')
ax[5].set_xlabel('time (ms)')

sag_ratio = np.mean((min_membr_pot-steady_state)/min_membr_pot)
avg_tau_ms = np.mean(tau_ms)

ephys_measures['tau'] = avg_tau_ms
ephys_measures['sag'] = np.mean((min_membr_pot-steady_state)/min_membr_pot)

ax[4].set_axis_off()

for a in ax:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.show()
fig.savefig(os.path.join(figpth, 'inhibitory_experiments.png'), dpi=300)
fig.savefig(os.path.join(figpth, 'inhibitory_experiments.pdf'), dpi=300)
fig.savefig(os.path.join(figpth, 'inhibitory_experiments.svg'), dpi=300)

plt.close('all')

## Plot comparison to Allen

measures = ['RMP', 'sag', 'mean_isi', 'first_isi', 'upstroke_downstroke', 'ap_width', 'ap_threshold', 'FR_slope', 'tau']
label = ['resting potential (mV)', 'sag ratio', 'mean isi (ms)', 'first isi (ms)', 'upstroke/downstroke', 'AP width (ms)', 'AP threshold (mV)', 'FI slope (spikes/pA)', r'$\tau_{m}$']
ylims = [(-85,-50), (-0.1,.4), (0, 450), (0, 450), (1, 6), (0, 2), (-75, -25), (0, 0.5), (0, 65)]
fig, ax = plt.subplots(2,5, figsize=(22,6))
cm_neurons = np.array([[218,166,156], [164,30,79], [119,161,187],[92,114,183]])/255

ax = ax.ravel()
for i,m in enumerate(measures):

    data = inh_allen[m].dropna().values
    v = ax[i].violinplot(data, vert=True, showmeans=False,showmedians=True)

    ax[i].plot(1., ephys_measures[m], 'o', color=cm_neurons[2], markersize=8)
    ax[i].set_ylabel(label[i])
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

    for pc in v['bodies']:
        pc.set_facecolor('black')
        pc.set_alpha(.2) 
    v['cbars'].set_color('black')
    v['cmaxes'].set_color('black')
    v['cmins'].set_color('black')
    v['cmedians'].set_color('black')

fig.tight_layout()
fig.savefig(figpth+'inhibitory_cell_vs_allen.png')
fig.savefig(figpth+'inhibitory_cell_vs_allen.svg')