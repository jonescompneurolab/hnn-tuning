""""Utils for connectivity tuning figures."""

import matplotlib.pyplot as plt
import numpy as np

from hnn_core import duecker_ET_model, simulate_dipole, jones_2009_model, MPIBackend
from hnn_core.cell import _get_gaussian_connection

from hnn_core.params import compare_dictionaries

from hnn_core.viz import plot_cell_morphology

# import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams['pdf.fonttype'] = 'truetype'
mainpth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/'
pathfig = os.path.join(mainpth, 'new_model_HNN', 'figures', 'conn')
os.makedirs(pathfig, exist_ok=True)
pathpaper = os.path.join(mainpth, 'methods_paper')

import sys
sys.path.append(os.path.join(mainpth, 'utils'))
from l5_plot_utils import get_l5pyr
from general_plot_utils import assign_color, get_cmap_neurons

# colormap settings
from matplotlib import colormaps
import matplotlib.patches as mpatches

cmap = colormaps['Set2']

cell_types = ['L2_inhibitory', 'L2_pyramidal', 'L5_inhibitory', 'L5_pyramidal']
cell_type_colors = {'L2_basket': cmap.colors[0], 'L2_pyramidal': cmap.colors[1], 
               'L5_basket': cmap.colors[3], 'L5_pyramidal': cmap.colors[4]}


# Hack a colormap

from scipy.interpolate import RectBivariateSpline
from hnn_core.extracellular import _get_laminar_z_coords, calculate_csd2d
from hnn_core.viz import plot_laminar_csd

def conn_space_constant(x, lamtha):
    """Explore effect of space constant on spike raster."""
    inplane_distance = 1  # example value
    A = 0.1
    return A*np.exp(-(x / (lamtha * inplane_distance)) ** 2)


def plot_connectivity_space_constant(net, lamtha1=6.125, lamtha2=6.125*0.8):

    sources = ['L5_pyramidal', 'L5_basket', 'L2_pyramidal', 'L2_basket', ]
    E_receptors = ['ampa', 'nmda']
    I_receptors = ['gabaa', 'gabab']

    connectivity_data = dict()

    for source in sources:
        connectivity_data[source] = dict()
        for conn in net.connectivity:
            if conn['src_type'] == source:
                t_type = conn['target_type']
                check_t = connectivity_data[source].get(t_type)

                if check_t is None:
                    connectivity_data[source][t_type] = dict()       
                rec = conn['receptor']
                connectivity_data[source][t_type][rec] = conn['nc_dict']['A_weight']


    conn_im = np.zeros((len(sources), len(sources), 2))

    xticklabels = dict()
    for s, source in enumerate(sources):
        xticks_source = []
        for t, target in enumerate(sources):
            
            check_t = connectivity_data[source].get(target)
            if 'basket' in source and check_t is not None:
                for i, rec in enumerate(I_receptors):
                    if rec in connectivity_data[source][target]:
                        conn_im[s, t, i] = -connectivity_data[source][target][rec]
                    xticks_source.append(f'{target}_{rec}')
            elif 'pyramidal' in source and check_t is not None:
                for i, rec in enumerate(E_receptors):
                    if rec in connectivity_data[source][target]:
                        conn_im[s, t, i] = connectivity_data[source][target][rec]
                    xticks_source.append(f'{target}_{rec}')
        xticklabels[source] = xticks_source

    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(1, 2,figsize=(16, 4),  width_ratios=[1, 0.5])
    norm = TwoSlopeNorm(vmin=np.min(conn_im), vmax=np.round(np.max(conn_im),4), vcenter=0)
    im = ax[0].pcolormesh(conn_im.reshape(len(sources), len(sources)*2), cmap='RdBu_r', norm=norm)
    ax[0].set_yticks(np.arange(.5, 4.5, 1))
    ax[0].set_yticklabels(sources, rotation='vertical', va='center')
    ax[0].set_xticks(np.arange(0.5, 8.5,1))
    ax[0].set_xticklabels(xticklabels['L2_pyramidal'])
    cb = plt.colorbar(im)
    cb.set_ticks([np.min(conn_im), 0, np.round(np.max(conn_im),4)])
    cb.set_ticklabels([f'{np.min(conn_im+0.1):.4f}', '0', f'{np.round(np.max(conn_im),4):.4f}'])
    cb.set_label('synaptic weight ($mu S$)')

    # space constant
    x = np.linspace(0, 300, 1000)
    y = conn_space_constant(x, lamtha1*30)
    y2 = conn_space_constant(x, lamtha2*30)
    ax[1].plot(x, 0.1*np.exp(-x**2/(2*129.66**2)), label='gauss with sigma=129.66, Campagnola, 2022', color='k')
    ax[1].plot(x, y, linestyle='--', label='conn with space constant lamtha=6.125*d', color='#88e788')
    ax[1].plot(x, y2, label='space constant lamtha=6.125*d*0.8', color='#198450')
    ax[1].legend()
    ax[1].set_xlabel('Distance (um)')
    ax[1].set_ylabel('Connection weight')
    ax[1].set_xlim([0, 300])
    ax[1].set_ylim([0, 0.11])

    return fig

def plot_conn_heatmap(net):

    # Make heatmaps
    exc_cells = ['L2_pyramidal', 'L5_pyramidal']
    inh_cells = ['L2_basket', 'L5_basket']

    # E-E
    pos_l5 = np.zeros((100, 3))
    for p, pos in enumerate(net.pos_dict['L5_pyramidal']):
        pos_l5[p] = np.array(pos)
        
    l5_gid = 170
    pos_dict_idx = int(l5_gid)-170
    ref_pos = net.pos_dict['L5_pyramidal'][pos_dict_idx]
    conn = net.connectivity[2]
    l5_center_conn_bl = np.zeros((10,10))
    for pos in net.pos_dict['L5_pyramidal']:
        target_pos = pos
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        l5_center_conn_bl[x_idx, y_idx] = weight

    ref_pos = (4, 4, 0)
    l5_center_conn_c = np.zeros((10,10))
    for pos in net.pos_dict['L5_pyramidal']:
        target_pos = pos
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        l5_center_conn_c[x_idx, y_idx] = weight

    l5_gid = 269
    pos_dict_idx = int(l5_gid)-170
    ref_pos = net.pos_dict['L5_pyramidal'][pos_dict_idx]
    conn = net.connectivity[2]
    l5_center_conn_tr = np.zeros((10,10))
    for pos in net.pos_dict['L5_pyramidal']:
        target_pos = pos
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        l5_center_conn_tr[x_idx, y_idx] = weight


    
    # I-I
    pos_l5 = np.zeros((100, 3))
    for p, pos in enumerate(net.pos_dict['L5_basket']):
        pos_l5[p] = np.array(pos)
        
    inh_gid = 135
    pos_dict_idx = int(inh_gid)-135
    ref_pos = net.pos_dict['L5_basket'][pos_dict_idx]
    conn = net.connectivity[2]
    inh_center_conn_bl = np.zeros((10,10))
    for pos in net.pos_dict['L5_basket']:
        target_pos = pos
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        inh_center_conn_bl[x_idx, y_idx] = weight

    inh_gid = 150
    pos_dict_idx = int(inh_gid)-135
    ref_pos = net.pos_dict['L5_basket'][pos_dict_idx]
    inh_center_conn_c = np.zeros((10,10))
    for pos in net.pos_dict['L5_basket']:
        target_pos = pos
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        inh_center_conn_c[x_idx, y_idx] = weight

    inh_gid = 169
    pos_dict_idx = int(inh_gid)-135
    ref_pos = net.pos_dict['L5_basket'][pos_dict_idx]
    conn = net.connectivity[2]
    inh_center_conn_tr = np.zeros((10,10))
    for pos in net.pos_dict['L5_basket']:
        target_pos = pos
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        inh_center_conn_tr[x_idx, y_idx] = weight


    # E-I
        
    l5_gid = 170
    pos_dict_idx = int(l5_gid)-170
    ref_pos = net.pos_dict['L5_pyramidal'][pos_dict_idx]
    conn = net.connectivity[2]
    l5_center_conn_bl_ei = np.zeros((10,10))
    for pos in net.pos_dict['L5_basket']:
        target_pos = pos
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        l5_center_conn_bl_ei[x_idx, y_idx] = weight

    ref_pos = (4, 4, 0)
    l5_center_conn_c_ei = np.zeros((10,10))
    for pos in net.pos_dict['L5_basket']:
        target_pos = pos
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        l5_center_conn_c_ei[x_idx, y_idx] = weight

    l5_gid = 269
    pos_dict_idx = int(l5_gid)-170
    ref_pos = net.pos_dict['L5_pyramidal'][pos_dict_idx]
    conn = net.connectivity[2]
    l5_center_conn_tr_ei = np.zeros((10,10))
    for pos in net.pos_dict['L5_basket']:
        target_pos = pos
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        l5_center_conn_tr_ei[x_idx, y_idx] = weight


    fig, ax = plt.subplots(3,3,figsize=(10, 10), sharex=True, sharey=True)
    ax[0,0].imshow(l5_center_conn_bl, origin='lower', cmap='hot')
    ax[0,1].imshow(l5_center_conn_c, origin='lower', cmap='hot')
    ax[0,2].imshow(l5_center_conn_tr, origin='lower', cmap='hot')

    ax[1,0].imshow(l5_center_conn_bl_ei, origin='lower', cmap='hot')
    ax[1,1].imshow(l5_center_conn_c_ei, origin='lower', cmap='hot')
    ax[1,2].imshow(l5_center_conn_tr_ei, origin='lower', cmap='hot')

    ax[2,0].imshow(inh_center_conn_bl, origin='lower', cmap='hot')
    ax[2,1].imshow(inh_center_conn_c, origin='lower', cmap='hot')
    im = ax[2,2].imshow(inh_center_conn_tr, origin='lower', cmap='hot')
    cb = fig.colorbar(im, ax=ax, shrink=0.1, fraction=0.1, aspect=7, label='syn. weight', orientation='horizontal')
    cb.set_ticks([0, 0.00025])
    cb.set_ticklabels([f'0', 'max'])
    ax[0,0].set_xlim([-.5, 9.5])
    ax[0,0].set_ylim([-.5, 9.5])
    ax[0,0].set_xticks([0, 9])
    ax[0,0].set_yticks([0, 9])
    ax[0,0].set_xticklabels([1, 10])
    ax[0,0].set_yticklabels([1, 10])

    return fig


def plot_interlayer_conn():

    fig, ax = plt.subplots(3,2, figsize=(8, 9), height_ratios=[0.05, 1,1], sharex=True)

    ax = ax.flatten()
    cell_types = ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal']
    for i,c in enumerate(cell_types):
        # stimulate L2/3 pyramidal
        net = duecker_ET_model()

        net.set_cell_positions(inplane_distance=30.)

        ampl = 3.
        t0 = 10

        net.add_tonic_bias(cell_type=c,section='soma',bias_name='tonic_soma',amplitude=ampl,t0=t0,tstop=t0+3)

        with MPIBackend(n_procs=10, mpi_cmd='mpiexec'):
            dpl = simulate_dipole(net, tstop=30, n_trials=1)

        # clamp signal for plotting
        timevec = net.cell_response.times

        clamp = np.zeros_like(timevec) 

        for bias in net.external_biases:
            for cell_type in net.external_biases[bias]:
                t0 = np.argmin(np.abs(timevec - net.external_biases[bias][cell_type]['t0']))
                tstop = np.argmin(np.abs(timevec - net.external_biases[bias][cell_type]['tstop']))

                clamp[t0:tstop] = net.external_biases[bias][cell_type]['amplitude']
        
        
        if i < 2:
            ax[i].plot(timevec, clamp, color='k', linewidth=1.5)
            ax[i].set_ylabel('clamp (nA)')
            ax[i].set_title(f'{cell_types[i]}')
            ax[i].set_xlim([5, 25])
            ax[i].set_axis_off()

        I = np.array(net.cell_response.spike_gids[0]) <270
        gids = np.array(net.cell_response.spike_gids).reshape(-1,)[I]
        times = np.array(net.cell_response.spike_times).reshape(-1,)[I]
        cell_types = ['L2_inhibitory', 'L2_pyramidal', 'L5_inhibitory', 'L5_pyramidal']
        cell_type_colors = {'L2_basket': cmap.colors[0], 'L2_pyramidal': cmap.colors[1], 
                    'L5_basket': cmap.colors[2], 'L5_pyramidal': cmap.colors[3]}

        colors = [assign_color(net, gid) for gid in gids]

        ax[i+2].scatter(times, gids, s=8, alpha=1., c=colors, marker='|')
        ax[i+2].set_ylabel('neuron index')
        ax[i+2].set_xlim([5, 25])
        ax[i+2].set_xticks(np.arange(5, 26, 10))
        ax[i+2].set_xlabel('time (ms)')
        ax[i+2].set_ylim([0, 270])
        ax[i+2].set_yticks([0, 35, 135, 170, 270])
        ax[i+2].invert_yaxis()

    for i in range(3):
        ax[i+2].set_ylim([ax[i+2].get_ylim()[0],ax[i+2].get_ylim()[1]])

    fig.tight_layout()
    return fig
    


# Find rheobase

def find_basal_rheo(model='jones'):

    #fig, ax = plt.subplots(2,2, figsize=(12, 10))
    # ax = ax.ravel()

    ampl = np.linspace(0.01, 1., 80)
    rheo = np.zeros((4,))

    for a in ampl: 
        if model == 'jones':
            net = jones_2009_model(mesh_shape=(1,1))
        else:
            net = duecker_ET_model(mesh_shape=(1,1))

        net.clear_connectivity()

        net.add_tonic_bias(cell_type='L2_basket', amplitude=a, t0=10., tstop=100.)
        net.add_tonic_bias(cell_type='L5_basket', amplitude=a, t0=10., tstop=100.)

        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=a, t0=10., tstop=100., section='basal_2', bias_name='bias_1')
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=a, t0=10., tstop=100., section='basal_3', bias_name='bias_2')
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=a, t0=10., tstop=100., section='apical_oblique', bias_name='bias_3')
        net.add_tonic_bias(cell_type='L5_pyramidal', amplitude=a, t0=10., tstop=100., section='basal_1', bias_name='bias_1')
        net.add_tonic_bias(cell_type='L5_pyramidal', amplitude=a, t0=10., tstop=100., section='basal_2', bias_name='bias_2')
        net.add_tonic_bias(cell_type='L5_pyramidal', amplitude=a, t0=10., tstop=100., section='apical_oblique', bias_name='apical_oblique')

        dpl = simulate_dipole(net, tstop=100, n_trials=1, dt =0.01, record_vsec='soma')

        for i in range(4):

            if i == 2:
                rheo[i] == rheo[0]
            if rheo[i] == 0:
                if any((np.array(net.cell_response.spike_gids[0]) == i)):
                    rheo[i] = a
                    #ax[i].plot(net.cell_response.times, net.cell_response.vsec[0][i]['soma'])
                    break

    return rheo

def rheo_jones_vs_duecker():

    rheo_jones = find_basal_rheo(model='jones')
    # find rheobases for new cells
    rheo_ca = find_basal_rheo(model='duecker')

    ## rheobase plots
    cell_types = ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal']

    jonnet = jones_2009_model(mesh_shape=(1,1))
    jonnet.clear_connectivity()

    canet = duecker_ET_model(mesh_shape=(1,1))
    canet.clear_connectivity()

    for i,c in enumerate(cell_types):
        if 'basket' in c:
            jonnet.add_tonic_bias(cell_type=c, amplitude=rheo_jones[i], t0=10., tstop=100., section='soma')
            canet.add_tonic_bias(cell_type=c, amplitude=rheo_ca[i], t0=10., tstop=100., section='soma')
        else:
            jonnet.add_tonic_bias(cell_type=c, amplitude=rheo_jones[i], t0=10., tstop=100., section='basal_1', bias_name='bias_1')
            canet.add_tonic_bias(cell_type=c, amplitude=rheo_ca[i], t0=10., tstop=100., section='basal_1', bias_name='bias_1')
            jonnet.add_tonic_bias(cell_type=c, amplitude=rheo_jones[i], t0=10., tstop=100., section='basal_2', bias_name='bias_2')
            canet.add_tonic_bias(cell_type=c, amplitude=rheo_ca[i], t0=10., tstop=100., section='basal_2', bias_name='bias_2')
            jonnet.add_tonic_bias(cell_type=c, amplitude=rheo_jones[i], t0=10., tstop=100., section='apical_oblique', bias_name='apical_oblique')
            canet.add_tonic_bias(cell_type=c, amplitude=rheo_ca[i], t0=10., tstop=100., section='apical_oblique', bias_name='apical_oblique')
    # simulate dipole
    dpl_jon = simulate_dipole(jonnet, tstop=100, n_trials=1, record_vsec='soma')
    dpl_ca = simulate_dipole(canet, tstop=100, n_trials=1, record_vsec='soma')

    fig, ax = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True)

    cell_types = ['L2b', 'L2p', 'L5b', 'L5p']
    for i in range(3):
        if i == 2:
            c = i+1
        else:
            c = i
        ax[0,i].plot(jonnet.cell_response.times, jonnet.cell_response.vsec[0][c]['soma'], color='k')
        ax[1,i].plot(jonnet.cell_response.times, canet.cell_response.vsec[0][c]['soma'], color='k')
        ax[0,i].set_title(f'{cell_types[c]}: {rheo_jones[c]:.4f} nA')
        ax[1,i].set_title(f'{cell_types[c]}: {rheo_ca[c]:.4f} nA')
        ax[1,i].set_xlabel('time (ms)')
        ax[0,0].set_ylabel('membrane potential ($\mu$V)')
        ax[1,0].set_ylabel('membrane potential ($\mu$V)')
        ax[0,i].set_ylim(-100, 50)
        ax[1,i].set_ylim(-100, 50)
        ax[0,i].set_xlim(0, 100)
        ax[0,i].set_xticks([0, 50, 100])

    fig.tight_layout()

    return fig


def plot_nmda_vs_exp2syn():

    from neuron import h
    fig, ax = plt.subplots(1, 5, figsize=(16,2.5), width_ratios=[0.5, 0.5, 0.5, 0.5, 1], constrained_layout=True)

    # NMDA current comparison without spiking

    # pre-synapse 
    soma_pre = h.Section(name='soma')
    soma_pre.L = soma_pre.diam = 12
    soma_pre.Ra = 100
    soma_pre.cm = 1
    soma_pre.insert('hh') 

    # clamp
    stim = h.IClamp(soma_pre(0.5))
    stim.delay = 20
    stim.dur = 2    
    stim.amp = 0.5

    # post-synapse
    basal_post = h.Section(name='basal')
    basal_post.L = 255
    basal_post.diam = 8.5
    basal_post.Ra = 100
    basal_post.cm = 1
    basal_post.insert('pas')

    syn = h.Exp2Syn(basal_post(0.5))
    syn.tau1 = 1
    syn.tau2 = 20
    syn.e = 0.0

    nc = h.NetCon(soma_pre(0.5)._ref_v, syn, sec=soma_pre)
    nc.weight[0] = 0.1
    nc.delay = 1

    t = h.Vector().record(h._ref_t)
    exp2curr = h.Vector().record(syn._ref_i)

    h.finitialize(-65)
    h.continuerun(500)

    h('forall delete_section()')  # clean up sections

    del stim, syn, nc, soma_pre, basal_post


    ## NMDA GAO

    # pre-synapse 
    soma_pre = h.Section(name='soma')
    soma_pre.L = soma_pre.diam = 12
    soma_pre.Ra = 100
    soma_pre.cm = 1
    soma_pre.insert('hh') 

    # clamp
    stim = h.IClamp(soma_pre(0.5))
    stim.delay = 20
    stim.dur = 2    
    stim.amp = 0.5

    # post-synapse
    basal_post = h.Section(name='basal')
    basal_post.L = 255
    basal_post.diam = 8.5
    basal_post.Ra = 100
    basal_post.cm = 1
    basal_post.insert('pas')

    syn = h.NMDA_gao(basal_post(0.5))

    nc = h.NetCon(soma_pre(0.5)._ref_v, syn, sec=soma_pre)
    nc.weight[0] = 0.1
    nc.delay = 1

    t = h.Vector().record(h._ref_t)
    gaocurr = h.Vector().record(syn._ref_i)

    h.finitialize(-65)
    h.continuerun(500)


    ax[0].plot(t.as_numpy(), np.abs(exp2curr.as_numpy()), color='k', alpha=1)
    ax2 = ax[0].twinx()
    ax2.plot(t.as_numpy(), np.abs(gaocurr.as_numpy()), color='r', alpha=0.75)


    h('forall delete_section()')  # clean up sections
    del stim, syn, nc, soma_pre, basal_post
    
    l5pyr = get_l5pyr(add_syn='Exp2Syn', syn_loc='basal_2')
    l5pyr.syn.e 
    l5pyr.syn.tau1 = 0.5
    l5pyr.syn.tau2 = 5.0
    soma = h.Section(name='soma')
    soma.L = soma.diam = 12
    soma.Ra = 100
    soma.cm = 1
    soma.insert('hh')

    nc = h.NetCon(soma(0.5)._ref_v, l5pyr.syn, sec=soma)
    nc.weight[0] = 0.2
    nc.delay = 1

    stim = h.IClamp(soma(0.5))
    stim.delay = 20
    stim.dur = 2    
    stim.amp = 0.5

    t = h.Vector().record(h._ref_t)
    v_pre = h.Vector().record(soma(0.5)._ref_v)
    v_post_basal = h.Vector().record(l5pyr._nrn_sections['basal_2'](0.5)._ref_v)
    v_post_soma = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)
    nmda_curr = h.Vector().record(l5pyr.syn._ref_i)
    ca_curr = h.Vector().record(l5pyr._nrn_sections['basal_2'](0.5)._ref_ica)
    ca_con = h.Vector().record(l5pyr._nrn_sections['basal_2'](0.5)._ref_cai)

    h.finitialize()
    h.continuerun(80)

    ax[1].plot(t, v_pre, color='#a52051', label='pre soma')
    ax[1].plot(t, v_post_soma, color='#436aa1', label='post soma', alpha=0.75)
    ax[1].plot(t, v_post_basal, color='#436aa1', label='post basal 2')
    ax[1].set_ylabel('Membrane potential (mV)')
    ax[1].legend()

    # calcium concentration
    ax[3].plot(t, ca_con, color='k')
    ax[3].set_ylabel('Ca2+ concentration (mM)')

    
    # voltage clamp
    bamps = np.linspace(-60, 0, 5)
    t1 = 11
    t2 = 34
    RMP = []
    nmda_grad = []
    for b,bamp in enumerate(bamps):

        h('forall delete_section()')  # clean up sections
        del nc, stim, l5pyr, soma
    
        l5pyr = get_l5pyr(add_syn='Exp2Syn', syn_loc='basal_2')
        l5pyr.syn.e 
        l5pyr.syn.tau1 = 0.5
        l5pyr.syn.tau2 = 5.0
        soma = h.Section(name='soma')
        soma.L = soma.diam = 12
        soma.Ra = 100
        soma.cm = 1
        soma.insert('hh')

        nc = h.NetCon(soma(0.5)._ref_v, l5pyr.syn, sec=soma)
        nc.weight[0] = 0.05
        nc.delay = 1

        stim = h.IClamp(soma(0.5))
        stim.delay = 20
        stim.dur = 2    
        stim.amp = 0.5

        vclamp = h.VClamp(l5pyr._nrn_sections['basal_2'](0.5))
        vclamp.dur[0] = 10
        vclamp.amp[0] = -80
        vclamp.dur[1] = 20
        vclamp.amp[1] = bamp 

        t = h.Vector().record(h._ref_t)
        v_pre = h.Vector().record(soma(0.5)._ref_v)
        v_post_basal = h.Vector().record(l5pyr._nrn_sections['basal_2'](0.5)._ref_v)
        v_post_soma = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)
        nmda_curr = h.Vector().record(l5pyr.syn._ref_i)
        ca_curr = h.Vector().record(l5pyr._nrn_sections['basal_2'](0.5)._ref_ica)
        ca_con = h.Vector().record(l5pyr._nrn_sections['basal_2'](0.5)._ref_cai)

        h.finitialize()
        h.continuerun(80)

        idx = np.where((t.as_numpy() >= 29) & (t.as_numpy() < 100))[0][0]
        nmda_grad = np.abs(np.min(nmda_curr.as_numpy()[idx:]))
        ax[4].plot(bamp, nmda_grad, 'o', color='k')
        ax[4].set_xlabel('basal membrane pot. (mV)')
        ax[4].set_ylabel('NMDA current (nA)')

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)


    h('forall delete_section()')  # clean up sections
    del nc, stim, l5pyr, soma

    l5pyr = get_l5pyr(add_syn='NMDA_gao', syn_loc='basal_2')
    soma = h.Section(name='soma')
    soma.L = soma.diam = 12
    soma.Ra = 100
    soma.cm = 1
    soma.insert('hh')

    nc = h.NetCon(soma(0.5)._ref_v, l5pyr.syn, sec=soma)
    nc.weight[0] = 0.2
    nc.delay = 1

    stim = h.IClamp(soma(0.5))
    stim.delay = 20
    stim.dur = 2    
    stim.amp = 0.5

    t = h.Vector().record(h._ref_t)
    v_pre = h.Vector().record(soma(0.5)._ref_v)
    v_post_basal = h.Vector().record(l5pyr._nrn_sections['apical_tuft'](0.5)._ref_v)
    v_post_soma = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)
    nmda_curr = h.Vector().record(l5pyr.syn._ref_iNMDA)
    ca_curr = h.Vector().record(l5pyr._nrn_sections['apical_tuft'](0.5)._ref_ica)
    ca_con = h.Vector().record(l5pyr._nrn_sections['apical_tuft'](0.5)._ref_cai)

    h.finitialize()
    h.continuerun(80)

    ax[2].plot(t, v_pre, color='#a52051', label='pre soma')
    ax[2].plot(t, v_post_soma, color='#436aa1', label='post soma', alpha=0.75)
    ax[2].plot(t, v_post_basal, color='#436aa1', label='post basal 2')
    ax[2].set_ylabel('Membrane potential (mV)')
    ax[2].legend()
    # calcium concentration
    ax[3].plot(t, ca_con, color='r', alpha=0.75)
    ax[3].set_ylabel('Ca2+ concentration (mM)')


    # voltage clamp
    bamps = np.linspace(-60, 0, 5)
    t1 = 11
    t2 = 34
    RMP = []
    nmda_grad = []
    for b,bamp in enumerate(bamps):
        h('forall delete_section()')  # clean up sections
        del nc, stim, l5pyr, soma
        l5pyr = get_l5pyr(add_syn='NMDA_gao', syn_loc='basal_2')

        soma = h.Section(name='soma')
        soma.L = soma.diam = 12
        soma.Ra = 100
        soma.cm = 1
        soma.insert('hh')

        nc = h.NetCon(soma(0.5)._ref_v, l5pyr.syn, sec=soma)
        nc.weight[0] = 0.05
        nc.delay = 1

        stim = h.IClamp(soma(0.5))
        stim.delay = 20
        stim.dur = 2    
        stim.amp = 0.5

        vclamp = h.VClamp(l5pyr._nrn_sections['basal_2'](0.5))
        vclamp.dur[0] = 10
        vclamp.amp[0] = -80
        vclamp.dur[1] = 20
        vclamp.amp[1] = bamp 

        t = h.Vector().record(h._ref_t)
        v_pre = h.Vector().record(soma(0.5)._ref_v)
        v_post_basal = h.Vector().record(l5pyr._nrn_sections['basal_2'](0.5)._ref_v)
        v_post_soma = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)
        nmda_curr = h.Vector().record(l5pyr.syn._ref_i)
        ca_curr = h.Vector().record(l5pyr._nrn_sections['basal_2'](0.5)._ref_ica)
        ca_con = h.Vector().record(l5pyr._nrn_sections['basal_2'](0.5)._ref_cai)

        h.finitialize()
        h.continuerun(80)

        idx = np.where((t.as_numpy() >= 29) & (t.as_numpy() < 100))[0][0]
        nmda_grad = np.abs(np.min(nmda_curr.as_numpy()[idx:]))
        ax[4].plot(bamp, nmda_grad, 'o', color='r', alpha=0.75)
        ax[4].set_xlabel('basal membrane pot. (mV)')
        ax[4].set_ylabel('NMDA current (nA)')

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    # for a in ax[:-1]:
    #     a.set_xlim([0, 80])
    #     a.set_xticks([0, 20])

    ax[0].spines['left'].set_bounds([0, 1])
    ax[0].spines['bottom'].set_bounds([0, 200])
    ax2.spines['right'].set_bounds([0, 0.05])
    ax2.spines['bottom'].set_bounds([0, 200])
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_yticks([0.05])
    ax[0].set_yticks([1])
    ax[0].set_xticks([200])
    ax[3].set_ylim(5e-5, 5.007e-5)
    ax[-1].set_ylim(0, 1)
    ax[-1].set_xlim(-61, 1)

    for i in np.arange(1,3):
        ax[i].spines['left'].set_bounds([0, 50])
        ax[i].set_yticks([50])
        ax[i].spines['bottom'].set_bounds([0, 50])
        ax[i].set_xticks([50])

    ax[3].spines['left'].set_bounds([5e-5, 5.002e-5])
    ax[3].set_yticks([5.002e-5])
    ax[3].set_yticklabels(['.002e-5'])
    ax[3].spines['bottom'].set_bounds([0, 50])
    ax[3].set_xticks([50])

    return fig