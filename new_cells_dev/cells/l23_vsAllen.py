""""L2/3 vs Allen data"""

import neuron
from neuron import h
import matplotlib.pyplot as plt
from neuron.units import ms, mV
import numpy as np
pth_mod = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/hnn-core/hnn_core/mod'
pth_data = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/data/l23'
figpth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/new_cells_dev/figures/l23_tuning/'
# import os
# os.mkdir(figpth)
plt.rcParams['svg.fonttype'] = 'none'

import sys
sys.path.append('/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/utils')
from l23_plot_utils import get_l2pyr, get_colormap
from functools import partial
from scipy.signal import find_peaks
from hnn_core.viz import plot_cell_morphology

import os
import pandas as pd
from scipy import signal
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

import pickle
import pandas as pd

h.load_file('stdrun.hoc')

plt.rcParams['svg.fonttype'] = 'none'
# Set global font sizes
plt.rcParams['font.size'] = 10          # Default text size
plt.rcParams['axes.titlesize'] = 10      # Title size
plt.rcParams['axes.labelsize'] = 10      # Axis labels (xlabel, ylabel)
plt.rcParams['xtick.labelsize'] = 10    # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 10     # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 10     # Legend text

def exp_func(t, tau, A, C):
    return A * (1 - np.exp(-t / tau)) + C

"""Load Allen data"""
with open(os.path.join(pth_data,'ephys_l2.pkl'), 'rb') as f:
    l2_allen = pickle.load(f)

with open(os.path.join(pth_data,'ephys_l3.pkl'), 'rb') as f:
    l3_allen = pickle.load(f)

ephys_measures = dict()


cm_neurons = get_colormap('inferno')[2]
cols = np.array(([27, 158, 119], [217,95,2], [117,112,179], [231,41,138], [102,166,30]))/255

plt.close('all')
## RMP
fig, ax = plt.subplots(2, 3, width_ratios=[0.5, 0.5, 1], figsize=(16, 6), constrained_layout=True)
ax = ax.flatten()

l2pyr = get_l2pyr()
 
section_names = ['soma', 'basal_1','basal_2', 'basal_3', 'apical_oblique', 'apical_trunk','apical_1', 'apical_tuft']
v_s = []
for s, section in enumerate(section_names):
    v_s.append(h.Vector().record(l2pyr._nrn_sections[section](0.5)._ref_v))

t = h.Vector().record(h._ref_t)
h.celsius = 37
h.finitialize(-65)
h.continuerun(100 * ms)
h.dt = 0.025

v_init = {}
for s, section in enumerate(section_names):
    if section == 'soma':
        soma_voltg = v_s[s].as_numpy()

    v = v_s[s].as_numpy()
    v_init[section] = v[-1]

print(v_init, flush=True)

ephys_measures['RMP'] = v_init['soma']

ax[0].plot(t,soma_voltg, color=cm_neurons[1], linewidth=2)
ax[0].plot(t.as_numpy()[-3],soma_voltg[-3],'o', color='k', markersize=9)
ax[0].set_ylabel('membrane potential (mV)')
ax[0].set_xlabel('time (ms)')

## Action potential
l2pyr = get_l2pyr(v_init=v_init)

stim = h.IClamp(l2pyr._nrn_sections['soma'](0.5))
stim.delay = 5
stim.dur = 3
stim.amp = 1.5
clamp_current = h.Vector().record(stim._ref_i)

t = h.Vector().record(h._ref_t)
v_soma = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)

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


ax[1].plot(t, v_soma, color=cm_neurons[1], linewidth=2)
ax[1].set_ylabel('membrane potential (mV)')
ax[1].plot(t[np.argmax(v_soma)], ap_peak, 'o', color=cols[0], markersize=9)
ax[1].set_ylim([-75, 50])
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
amp_isi = np.mean(np.concatenate((l2_allen['max_amp_isi'].dropna().values,l3_allen['max_amp_isi'].dropna().values)))*1.2
l2pyr = get_l2pyr(v_init=v_init)
stim = h.IClamp(l2pyr._nrn_sections['soma'](0.5))
stim.delay = 20
stim.dur = 1000
stim.amp = amp_isi

t = h.Vector().record(h._ref_t)
soma_v = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)

h.celsius = 37
h.finitialize()
h.continuerun(1020 * ms)

ax[2].plot(t,soma_v, color=cm_neurons[1], linewidth=2)
ax[2].set_xlim([0, 1000])
ax[2].set_yticks([-70,0, 50])
ax[2].set_ylim([-75, 50])
ax[2].set_xlabel('time (ms)')

aps = find_peaks(soma_v,height=0)[0]
ephys_measures['first_isi'] = np.diff(t.as_numpy()[aps])[0]
ephys_measures['mean_isi'] = np.mean(np.diff(t.as_numpy()[aps]))

## FR slope

l_stim = 1000

max_amp_isi = np.max(np.concatenate((l2_allen['max_amp_isi'].dropna().values,l3_allen['max_amp_isi'].dropna().values)))

ampl = np.array([amp_isi/3, amp_isi/1.5, amp_isi, max_amp_isi/1.5, max_amp_isi])

soma_v = np.zeros((len(ampl),int(1/h.dt*l_stim+1)))
fr = np.zeros_like(ampl)

for i,amp in enumerate(ampl):

    l2pyr= get_l2pyr()

    stim = h.IClamp(l2pyr._nrn_sections['soma'](0.5))
    stim.delay = 20
    stim.dur = 1000
    stim.amp = amp
    clamp_current = h.Vector().record(stim._ref_i)

    t = h.Vector().record(h._ref_t)
    v_soma = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)

    h.celsius = 37
    h.finitialize()
    h.continuerun((l_stim+stim.delay)* ms)

    pk = find_peaks(v_soma, height=0)

    fr[i] = len(pk[0])


ax[3].plot(ampl,fr, color=cm_neurons[1], linewidth=2)
ax[3].set_ylabel('FR (Hz)')
ax[3].set_xlabel('amplitude (nA)')
ax[3].set_xlim([ampl[0], ampl[-1]])
pA = 0.001 # nA (convert nA to pA)
ephys_measures['FR_slope'] = np.polyfit(ampl,fr,1)[0]
ephys_measures['FR_slope'] *= pA


# Hyperpolarization experiments
sag_stim_amp_l2 = l2_allen['sag_stim_amp'].dropna()
sag_stim_amp_l2  = np.concatenate(sag_stim_amp_l2.values)

sag_stim_amp_l3 = l3_allen['sag_stim_amp'].dropna()
sag_stim_amp_l3  = np.concatenate(sag_stim_amp_l3.values)

sag_stim_amp = np.linspace(-0.5, -0.1, 10) #np.unique(np.concatenate((sag_stim_amp_l2, sag_stim_amp_l3)))
tau_ms = np.zeros_like(sag_stim_amp)
steady_state = np.zeros_like(sag_stim_amp)
min_membr_pot = np.zeros_like(sag_stim_amp)

for i, amp in enumerate(sag_stim_amp):
    l2pyr= get_l2pyr()

    h.distance(sec = l2pyr._nrn_sections['soma'])

    stim = h.IClamp(l2pyr._nrn_sections['soma'](0.5))
    stim.delay = 50
    stim.dur = 500
    stim.amp = amp

    v = h.Vector().record(l2pyr._nrn_sections['soma'](0.5)._ref_v)

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

    ax[5].plot(t_np, v_np, color=cm_neurons[1], linewidth=2)
    idx = np.argmin(np.abs(t_np-tau_ms[i]))
    ax[5].plot(t_np[idx+start_idx], v_np[idx+start_idx], 'o', color='k', markersize=9)

ax[5].set_xlim([0, 650])
ax[5].set_ylabel('membrane potential (mV)')
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
fig.savefig(os.path.join(figpth, 'l23_experiments.png'), dpi=300)
fig.savefig(os.path.join(figpth, 'l23_experiments.pdf'), dpi=300)
fig.savefig(os.path.join(figpth, 'l23_experiments.svg'), dpi=300)

plt.close('all')

## Plot comparison to Allen

measures = ['RMP', 'sag', 'mean_isi', 'first_isi', 'upstroke_downstroke', 'ap_width', 'ap_threshold', 'FR_slope', 'tau']
label = ['resting potential (mV)', 'sag ratio', 'mean isi (ms)', 'first isi (ms)', 'upstroke/downstroke', 'AP width (ms)', 'AP threshold (mV)', 'FI slope (spikes/pA)', r'$\tau_{m}$']
ylims = [(-85,-50), (-0.1,.4), (0, 450), (0, 450), (1, 6), (0, 2), (-75, -25), (0, 0.5), (0, 65)]
fig, ax = plt.subplots(2,5, figsize=(22,6))
cm_neurons = np.array([[218,166,156], [164,30,79], [119,161,187],[92,114,183]])/255

ax = ax.ravel()
for i,m in enumerate(measures):

    data = [l2_allen[m].dropna(),l3_allen[m].dropna()]
    v = ax[i].violinplot(data, positions=[1.2,1.8], vert=True, showmeans=False,
                  showmedians=True)

    ax[i].plot(1.2, ephys_measures[m], 'o', color=cm_neurons[1], markersize=8)
    ax[i].plot(1.8, ephys_measures[m], 'o', color=cm_neurons[1], markersize=8)
    ax[i].set_ylabel(label[i])
    ax[i].set_xticks([1.2,1.8],labels=['L2', 'L3'])
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
fig.savefig(figpth+'l23_cell_vs_allen.png')
fig.savefig(figpth+'l23_cell_vs_allen.svg')