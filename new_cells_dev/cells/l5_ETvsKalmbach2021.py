""""Compare electrophysiological properties of L5 ET cell model to Kalmbach et al. 2021."""

import neuron
from neuron import h
import matplotlib.pyplot as plt
from neuron.units import ms
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import pickle
from scipy.fft import fft, fftfreq
from scipy.signal import chirp, windows, savgol_filter

pth_mod = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/local_hnn/hnn_core/mod'
figpth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/new_cells_dev/figures/l5_ET_tuning/'

import pandas as pd
import sys
sys.path.append('/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/utils')

from l5_plot_utils import get_model_params, get_l5pyr
from functools import partial
import os
from general_plot_utils import assign_color, get_cmap_neurons

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# Set global font sizes
plt.rcParams['font.size'] = 16          # Default text size
plt.rcParams['axes.titlesize'] = 16      # Title size
plt.rcParams['axes.labelsize'] = 16      # Axis labels (xlabel, ylabel)
plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 16     # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 16     # Legend text
cols = np.array(([27, 158, 119], [217,95,2], [117,112,179], [231,41,138], [102,166,30]))/255

pth_data = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/allen_data/l5'

cm_neurons = get_cmap_neurons()

# general L5 tau from Allen
with open(os.path.join(pth_data,'ephys_l5.pkl'), 'rb') as f:
    l5_allen = pickle.load(f)


# Kalmbach et al. 2021 data
df = pd.read_csv(os.path.join(pth_data, 'kalmbach_2021_l5_electrophys.csv'))
df['etype'].value_counts() # ET = 1
l5df = df[df['etype'] == 1]

model_params = get_model_params()


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8), constrained_layout=True)
gs = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[0.5, 0.5, 1])

# First row: 3 plots
ax_0_0 = fig.add_subplot(gs[0, 0])
ax_0_1 = fig.add_subplot(gs[0, 1])
ax_0_2 = fig.add_subplot(gs[0, 2])

# Second row: 3 plots
ax_1_0 = fig.add_subplot(gs[1, 0])
ax_1_1 = fig.add_subplot(gs[1, 1])
ax_1_2 = fig.add_subplot(gs[1, 2])  # This is where ax[0,3] content will go

# Third row: 2 plots
ax_2_0 = fig.add_subplot(gs[2, 0:2])  # This is where ax[1,2] content will go
ax_2_1 = fig.add_subplot(gs[2, 2])   # This is where ax[1,3] content will go

"""resting membrane potential"""
l5pyr = get_l5pyr(v_init={'soma': -65,
                'basal_1': -65,
                'basal_2': -65,
                'basal_3': -65,
                'apical_oblique': -65,
                'apical_trunk': -65,
                'apical_1': -65,
                'apical_2': -65,
                'apical_tuft': -65})
 
t = h.Vector().record(h._ref_t)
section_names = ['soma', 'basal_1','basal_2', 'basal_3', 'apical_oblique', 'apical_trunk','apical_1', 'apical_2', 'apical_tuft']
v_s = []
for s, section in enumerate(section_names):
    v_s.append(h.Vector().record(l5pyr._nrn_sections[section](0.5)._ref_v))



h.celsius = 37
h.finitialize()
h.continuerun(100 * ms)

v_init = {}
for s, section in enumerate(section_names):
    if section == 'soma':
        soma_voltg = v_s[s].as_numpy()

    v = v_s[s].as_numpy()
    v_init[section] = v[-1]

print(v_init, flush=True)

rmp = soma_voltg[-3]

ax_0_0.plot(t,soma_voltg, color=cm_neurons[3], linewidth=2)
ax_0_0.plot(t.as_numpy()[-3],soma_voltg[-3],'o', color='k', markersize=7)
ax_0_0.set_ylabel('membrane \n potential (mV)')
ax_0_0.set_xlabel('time (ms)')
ax_0_0.set_xlim([0, 105])
ax_0_0.set_xticks([0, 50, 100])
ax_0_0.set_ylim([-75, -65])
ax_0_0.set_yticks([-75, -65])

"""Action potential"""
l5pyr = get_l5pyr()
stim = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
stim.delay = 5
stim.dur = 3
stim.amp = 3
clamp_current = h.Vector().record(stim._ref_i)

t = h.Vector().record(h._ref_t)
v_soma = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)
iclamp = h.Vector().record(stim._ref_i)

h.celsius = 37
h.dt = .01
h.finitialize()
h.continuerun(100 * ms)

ap, properties = find_peaks(v_soma.as_numpy(), height=0, width=0)

v_grad = np.gradient(v_soma)
smooth_v = signal.savgol_filter(np.diff(v_soma.as_numpy), 101, 3)

ap_thresh_idx = np.where(smooth_v/np.diff(t.as_numpy()) > 20)[0][0]+1

ap_thresh = v_soma[ap_thresh_idx]

if len(ap) > 0:
    ap_peak = properties['peak_heights'][0]
else:
    ap_peak = v_soma[-1]

v2 = np.gradient(np.gradient(v_soma.as_numpy()))

ahp, props_v2 = find_peaks(-v2, width=1)

ap_hw = properties['widths'][0] * h.dt

v2 = np.gradient(np.gradient(smooth_v))

start = int(ap)
stop = int(np.where(t.as_numpy()>11)[0][0])

elbow_p = int(np.argmin(v_soma.as_numpy()[start:stop])+start)

f_AHP = v_soma.as_numpy()[elbow_p] - ap_thresh

ax_0_1.plot(t, v_soma, color=cm_neurons[3], linewidth=2)
ax_0_1.plot(t[np.argmax(v_soma)], ap_peak, 'o', color=cols[0], markersize=7)
ax_0_1.set_ylim([-80, 50])
ax_0_1.set_yticks([-70,0, 50])
ax_0_1.set_xlim([0, 20])
ax_0_1.plot(t[ap_thresh_idx], ap_thresh, 'o', color=cols[1], markersize=7)
ax_0_1.plot(t.as_numpy()[elbow_p], v_soma.as_numpy()[elbow_p], 'o', color=cols[3], markersize=7)
ax_0_1.set_xlabel('time (ms)')

"""burst ISI """
l5pyr = get_l5pyr()
stim = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
stim.delay = 20
stim.dur = 2000
stim.amp = .8
iclamp = h.Vector().record(stim._ref_i)
t = h.Vector().record(h._ref_t)
soma_voltg = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)
h.celsius = 37
h.finitialize()
h.continuerun(2000 * ms)

idx = np.argmin(np.abs(t.as_numpy()-100))
spk, prop = signal.find_peaks(soma_voltg.as_numpy()[:idx], height=0)

min_burst_isi = np.mean(np.diff(t.as_numpy()[spk]))
ax_0_2.plot(t, soma_voltg, color=cm_neurons[3], linewidth=2)
ax_0_2.set_xlim([0, 2000])
ax_0_2.set_xticks(np.arange(0, 2001, 500))
ax_0_2.set_ylim([-80, 50])
ax_0_2.set_yticks([-70,0, 50])
ax_0_2.set_xlabel('time (ms)')

"""FI slope - MOVED TO ax_1_2"""
del l5pyr 
h('forall delete_section()')
ampls = np.linspace(.05,2,20)

F = np.zeros_like(ampls)
for i,amp in enumerate(ampls):

    l5pyr= get_l5pyr()

    stim = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
    stim.delay = 1
    stim.dur = 1000
    stim.amp = amp

    t = h.Vector().record(h._ref_t)
    soma_voltg = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)

    h.celsius = 37
    h.finitialize()
    h.continuerun(1001 * ms)

    soma_voltg = soma_voltg.as_numpy()

    F[i] = len(signal.find_peaks(soma_voltg, height=0)[0])

FI_slope = np.diff(np.array(F))/np.diff(ampls)[0]
pA = 0.001
max_fi = max(FI_slope)*pA

ax_1_2.plot(ampls, F, color=cm_neurons[3], linewidth=2)
ax_1_2.set_ylabel('firing rate (Hz)')
ax_1_2.set_xlabel('current injection (nA)')
ax_1_2.set_xlim([0, 2])
ax_1_2.set_xticks(np.arange(0, 2.1, 0.5))
ax_1_2.set_ylim([0, 20])
ax_1_2.set_yticks(np.arange(0, 21, 10))

"""Response to chirp"""

l5pyr = get_l5pyr()

## Play stimulus
# set up clamp
stim = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
stim.delay = 0

# Define simulation parameters
h.tstop = 20_100
stim.dur = h.tstop
h.dt = 0.1

# Create a chirp stimulus (linear frequency sweep)
t_chirp = np.arange(0, stim.dur, h.dt)

f0 = 0.2
f1 = 40
A = .05

# logarithmic chirp
chirp_signal = A * chirp(t_chirp/1000, f0, 20, f1, method='log')

stim.amp = 0
# Apply the chirp stimulus using a Vector and play()
stim_vec = h.Vector(chirp_signal)
t = np.arange(0, stim.dur, h.dt)
time_vec = h.Vector(t)
stim_vec.play(stim._ref_amp, time_vec, 1)

# Record membrane potential
v_rec = h.Vector()
v_rec.record(l5pyr._nrn_sections['soma'](0.5)._ref_v)
t_rec = h.Vector().record(h._ref_t)
# Run the simulation
h.finitialize()
h.continuerun(h.tstop)

## Get impedance
hann_win = windows.hann(len(chirp_signal), len(chirp_signal)*1.)

N = len(v_rec)-1001
T = h.dt/1000
freq = fftfreq(N, T)[:N//2]
lowf = np.argmin(np.abs(freq-0.2))

fft_membr = fft(v_rec.as_numpy()[1:])
fft_chirp = fft(chirp_signal)

maxf = int(np.where(freq == 40)[0])

freq = freq[lowf:maxf]
Z = fft_membr[lowf:maxf]/fft_chirp[lowf:maxf]
zap = np.abs(Z[:N//2])

norm_zap = zap / zap.max()

res_freq = freq[np.argmax(zap)]
res_freq_idx = np.argmin(np.abs(freq - res_freq))
cutoff3db = np.sqrt(0.5*norm_zap.max())
cutoff3db_idx = np.argmin(np.abs(norm_zap - cutoff3db))
cutoff3db_freq = freq[cutoff3db_idx]

zap_smooth = savgol_filter(norm_zap, window_length=10, polyorder=1)  

ax_1_0.plot(t_rec, v_rec.as_numpy(), color=cm_neurons[3], linewidth=2)
ax_1_0.set_xlabel('time (ms)')
ax_1_0.set_ylabel('membrane \n potential (mV)')
ax_1_0.set_xlim([0,20000])
ax_1_0.set_xticks(np.arange(0, 20001, 10000))
ax_1_0.set_ylim([-76, -72])
ax_1_0.set_yticks([-76, -72])

ax_1_1.plot(freq, zap_smooth, color=cm_neurons[3], linewidth=2)
ax_1_1.plot(res_freq,zap_smooth[np.argmax(norm_zap)], 'o', color='k', markersize=7)
ax_1_1.plot(cutoff3db_freq,cutoff3db, 'o', color=cols[0], markersize=7)
ax_1_1.set_xlabel('frequency (Hz)')
ax_1_1.set_ylabel('amplitude')
ax_1_1.set_xlim([0, 40])
ax_1_1.set_xticks(np.arange(0, 41, 10))
ax_1_1.set_ylim([0, 1])
ax_1_1.set_yticks([0, 1])

## Rebound slope - MOVED TO ax_2_0

# get sag stim amplitudes from Allen data
sag_stim_amp = l5_allen['sag_stim_amp'].dropna()
sag_stim_amp = np.unique(np.concatenate(sag_stim_amp.values))

soma_v = np.zeros((len(sag_stim_amp),int(1/h.dt*1200+1)))
peak_heights = np.zeros_like(sag_stim_amp)
steady_state = np.zeros_like(sag_stim_amp)
min_membr_pot = np.zeros_like(sag_stim_amp)
sag_ratio = np.zeros_like(sag_stim_amp)
for i,amp in enumerate(sag_stim_amp):

    l5pyr= get_l5pyr()

    stim = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
    stim.delay = 20
    stim.dur = 1000
    stim.amp = amp
    clamp_current = h.Vector().record(stim._ref_i)

    t = h.Vector().record(h._ref_t)
    soma_voltg = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)

    h.celsius = 37
    h.finitialize()
    h.continuerun(1200 * ms)

    steady_idx = np.argmin(np.abs(t-1000))
    soma_v[i] = soma_voltg.as_numpy()

    pk = find_peaks(soma_v[i][steady_idx:], height=[-90, 50])
    if len(pk[0]) > 1:
        peak_heights[i] = pk[1]['peak_heights'][0]
    elif len(pk[0]) == 0:
        peak_heights[i] = np.nan
    else:
        peak_heights[i] = pk[1]['peak_heights']
    
    min_membr_pot[i] = np.min(soma_v[i])
    steady_state[i] = soma_v[i][steady_idx]

    ax_2_0.plot(t, soma_voltg, color=cm_neurons[3])
    if i == 0:
        ax_2_0.plot(t[np.argmin(soma_v[i])], min_membr_pot[i], 'o', color='k', markersize=7)
        ax_2_0.plot(t[steady_idx], steady_state[i], 'o', color=cols[0], markersize=7)
        ax_2_0.plot(t[int(steady_idx+pk[0])], peak_heights[i], 'o', color=cols[1], markersize=7)

ax_2_0.set_ylabel('membrane \n potential (mV)')
ax_2_0.set_xlabel('time (ms)')
ax_2_0.set_xlim([0, 1200])
ax_2_0.set_xticks(np.arange(0, 1201, 400))
ax_2_0.set_ylim([-85, -70])
ax_2_0.set_yticks([-85, -70])

ax_2_1.plot(steady_state, peak_heights, color=cm_neurons[3])
ax_2_1.set_xlabel('steady state (mV)')
ax_2_1.set_ylabel('depolarization \n potential (mV)')
ax_2_1.set_ylim([-74, -73])
ax_2_1.set_yticks([-74, -73])

rebound_slope = np.polyfit(steady_state, peak_heights,1)[0]

# Sag ratio
inp_res_max = np.polyfit(sag_stim_amp, min_membr_pot,1)[0]
inp_res_steady = np.polyfit(sag_stim_amp, steady_state,1)[0]

sag_ratio = inp_res_max/inp_res_steady

# Apply spine formatting to all axes
for ax in [ax_0_0, ax_0_1, ax_0_2, ax_1_0, ax_1_1, ax_1_2, ax_2_0, ax_2_1]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

#plt.tight_layout()
plt.show()

fig.savefig(os.path.join(figpth, 'l5cell_experiments.pdf'), dpi=300)
fig.savefig(os.path.join(figpth, 'l5cell_experiments.png'), dpi=300)
fig.savefig(os.path.join(figpth, 'l5cell_experiments.svg'), dpi=300)
plt.close('all')

fig, ax = plt.subplots(3,4, figsize=(12,8))

viol = [None]*12
ax = ax.ravel()
viol[0] = ax[0].violinplot(l5df['RMP'], vert=True, showmeans=False,
                  showmedians=True)
ax[0].set_yticks(np.arange(-80, -40, 10))
ax[0].plot(1, rmp, 'o', color=cm_neurons[3], markersize=8)
ax[0].set_xlabel('time (ms)')
ax[0].set_ylim([-90, -50])
ax[0].set_yticks([-90, -50])
ax[0].set_ylabel('resting potential (mV)')

viol[1] = ax[1].violinplot(l5df['AP_threshold'], vert=True, showmeans=False,
                  showmedians=True)
ax[1].plot(1, ap_thresh, 'o', color=cm_neurons[3], markersize=8)
ax[1].set_xlabel('time (ms)')
ax[1].set_ylim([-55, -25])
ax[1].set_yticks([-55, -25])
ax[1].set_ylabel('AP threshold (mV)')

viol[2] = ax[2].violinplot(l5df['AP_peak'], vert=True, showmeans=False,
                  showmedians=True)
ax[2].plot(1, ap_peak, 'o', color=cm_neurons[3], markersize=8)
ax[2].set_yticks([20, 80])
ax[2].set_ylabel('AP peak (mV)')
ax[2].set_xlabel('time (ms)')

viol[3] = ax[3].violinplot(l5df['AP_HW'], vert=True, showmeans=False,
                  showmedians=True)
ax[3].plot(1, ap_hw, 'o', color=cm_neurons[3], markersize=8)

ax[3].set_ylabel('AP half width (mV)')
ax[3].set_yticks([0.4, 1.0])


viol[4] = ax[4].violinplot(l5df['f_AHP'], vert=True, showmeans=False,
                  showmedians=True)
ax[4].plot(1, f_AHP, 'o', color=cm_neurons[3], markersize=8)

ax[4].set_ylabel('fast AHP (mV)')
ax[4].set_yticks([-30, 0])

viol[5] = ax[5].violinplot(l5df['Sag_ratio'], vert=True, showmeans=False,
                  showmedians=True)
ax[5].set_yticks([1, 1.6])
ax[5].set_ylim([1, 1.6])
ax[5].plot(1, sag_ratio, 'o', color=cm_neurons[3], markersize=8)

ax[5].set_ylabel('sag ratio')

viol[6] = ax[6].violinplot(l5df['Rebound'], vert=True, showmeans=False,
                  showmedians=True)
ax[6].plot(1,rebound_slope, 'o', color=cm_neurons[3], markersize=8)
ax[6].set_yticks([-0.5, 0])
ax[6].set_ylabel('rebound slope')

viol[7] = ax[7].violinplot(l5df['SSRn'], vert=True, showmeans=False,
                  showmedians=True)
ax[7].plot(1, inp_res_max, 'o', color=cm_neurons[3], markersize=8)

ax[7].set_ylabel('input resistance MOhms')
ax[7].set_yticks([10, 90])


viol[8] = ax[8].violinplot(l5df['FI_slope_max'], vert=True, showmeans=False,
                  showmedians=True)
ax[8].plot(1, max_fi, 'o', color=cm_neurons[3], markersize=8)
ax[8].set_ylabel('max gain(spikes/pA)')
ax[8].set_yticks([0.02, 0.1])


viol[9] = ax[9].violinplot(l5df['min_burst_ISI_5'][~np.isnan(l5df['min_burst_ISI_5'])], vert=True, showmeans=False,
                  showmedians=True)
ax[9].plot(1, min_burst_isi, 'o', color=cm_neurons[3], markersize=8)
ax[9].set_ylabel('burst ISI (ms)')
ax[9].set_yticks([0, 100])

viol[10] = ax[10].violinplot(l5df['Fr'], vert=True, showmeans=False,
                  showmedians=True)
ax[10].plot(1,res_freq, 'o', color=cm_neurons[3], markersize=8)
ax[10].set_yticks([1, 7])
ax[10].set_ylabel('resonance freq')

viol[11] = ax[11].violinplot(l5df['Cutoff'].values[np.isnan(l5df['Cutoff'].values)==False], vert=True, showmeans=False,showmedians=True)
ax[11].plot(1,cutoff3db_freq, 'o', color=cm_neurons[3], markersize=8)
ax[11].set_ylabel('3db cutoff')
ax[11].set_yticks([4, 16])
for v in viol:
    for pc in v['bodies']:
        pc.set_facecolor('black')
        pc.set_alpha(0.2)
        
    v['cbars'].set_color('black')
    v['cmaxes'].set_color('black')
    v['cmins'].set_color('black')
    v['cmedians'].set_color('black')

for a in ax:
    a.set_xticks([])
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['bottom'].set_visible(False)

fig.tight_layout()
plt.show()

fig.savefig(os.path.join(figpth, 'l5cell_vs_kalmbach2021.pdf'))
fig.savefig(os.path.join(figpth, 'l5cell_vs_kalmbach2021.svg'))

fig.savefig(os.path.join(figpth, 'l5cell_vs_kalmbach2021.png'))