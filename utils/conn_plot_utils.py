""""Utils for connectivity tuning figures."""

import matplotlib.pyplot as plt
import numpy as np
from hnn_core import duecker_ET_model, simulate_dipole, jones_2009_model, MPIBackend
from hnn_core.params import compare_dictionaries
from hnn_core.viz import plot_cell_morphology

# import pandas as pd
from scipy import signal
from matplotlib.gridspec import GridSpec
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mainpth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/'
pathfig = os.path.join(mainpth, 'new_model_HNN', 'figures', 'conn')
os.makedirs(pathfig, exist_ok=True)
pathpaper = os.path.join(mainpth, 'methods_paper')

import sys
sys.path.append(mainpth)
from general_plot_utils import assign_color, get_cmap_neurons

from scipy.interpolate import RectBivariateSpline
from hnn_core.extracellular import _get_laminar_z_coords, calculate_csd2d
from hnn_core.viz import plot_laminar_csd

## Function to find rheobase in the model

def find_basal_rheo(net):

    #fig, ax = plt.subplots(2,2, figsize=(12, 10))
    # ax = ax.ravel()

    ampl = np.linspace(0.01, 1., 80)
    rheo = np.zeros((4,))

    for a in ampl: 

        net.add_tonic_bias(cell_type='L2_basket', amplitude=a, t0=10., tstop=100.)
        net.add_tonic_bias(cell_type='L5_basket', amplitude=a, t0=10., tstop=100.)

        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=a, t0=10., tstop=100., section='basal_2', bias_name='bias_1')
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=a, t0=10., tstop=100., section='basal_3', bias_name='bias_2')
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=a, t0=10., tstop=100., section='apical_oblique', bias_name='bias_3')
        net.add_tonic_bias(cell_type='L5_pyramidal', amplitude=a, t0=10., tstop=100., section='basal_2', bias_name='bias_1')
        net.add_tonic_bias(cell_type='L5_pyramidal', amplitude=a, t0=10., tstop=100., section='basal_3', bias_name='bias_2')
        net.add_tonic_bias(cell_type='L5_pyramidal', amplitude=a, t0=10., tstop=100., section='apical_oblique', bias_name='apical_oblique')

        dpl = simulate_dipole(net, tstop=100, n_trials=1, dt =0.01, record_vsec='soma')

        for i in range(4):

            if i == 2:
                rheo[i] == rheo[0]
            if rheo[i] == 0:
                if any((np.array(net.cell_response.spike_gids[0]) == i)):
                    rheo[i] = a
                    break

    return rheo

def plot_rheo(net, rheo, ax):

    cell_types = ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal']

    for i,c in enumerate(cell_types):
        if 'basket' in c:
            net.add_tonic_bias(cell_type=c, amplitude=rheo[i], t0=10., tstop=100., section='soma')
        else:
            net.add_tonic_bias(cell_type=c, amplitude=rheo[i], t0=10., tstop=100., section='basal_2', bias_name='bias_1')
            net.add_tonic_bias(cell_type=c, amplitude=rheo[i], t0=10., tstop=100., section='basal_3', bias_name='bias_2')
            net.add_tonic_bias(cell_type=c, amplitude=rheo[i], t0=10., tstop=100., section='apical_oblique', bias_name='apical_oblique')
    # simulate dipole
    dpl = simulate_dipole(net, tstop=100, n_trials=1, record_vsec='soma')
    
    fig, ax = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True)

    cell_types = ['L2b', 'L2p', 'L5b', 'L5p']
    for i in range(3):
        if i == 2:
            c = i+1
        else:
            c = i
        ax[i].plot(net.cell_response.times, net.cell_response.vsec[0][c]['soma'], color='k')
        ax[i].set_title(f'{cell_types[c]}: {rheo[c]:.4f} nA')
        ax[i].set_xlabel('time (ms)')
        ax[i].set_ylabel('membrane potential ($\mu$V)')
        ax[i].set_ylim(-100, 50)
        ax[i].set_xlim(0, 100)
        ax[i].set_xticks([0, 50, 100])

    return ax