""""Utils for connectivity tuning figures."""

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from hnn_core import duecker_ET_model, simulate_dipole, jones_2009_model, MPIBackend
from hnn_core.cell import _get_gaussian_connection

from hnn_core.params import compare_dictionaries

from hnn_core.viz import plot_cell_morphology

from neuron import h
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

    targets = ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal'  ]
    src_inh = ['L2_basket', 'L5_basket']
    src_pyr = ['L2_pyramidal', 'L5_pyramidal']
    E_receptors = ['ampa', 'nmda']
    I_receptors = ['gabaa', 'gabab']

    connectivity_data = dict()

    for src in targets:
        connectivity_data[src] = dict()
        
        for conn in net.connectivity:
            if conn['src_type'] == src:
                t_type = conn['target_type']
                check_t = connectivity_data[src].get(t_type)
                if check_t is None:
                    connectivity_data[src][t_type] = dict()       
                rec = conn['receptor']
                connectivity_data[src][t_type][rec] = conn['nc_dict']['A_weight']


    conn_im_pyr = np.zeros((len(src_pyr), len(targets), 2))
    conn_im_inh = np.zeros((len(src_inh), len(targets), 2))

    xticklabels_pyr = []

    for t, target in enumerate(targets):
        for i, rec in enumerate(E_receptors):
              xticklabels_pyr.append(f'{target}_{rec}')

    for s, src in enumerate(src_pyr):
        for t, target in enumerate(targets):   
            check_t = connectivity_data[src].get(target)
            
            if check_t is not None:
                for i, rec in enumerate(E_receptors):
                    if rec in connectivity_data[src][target]:
                        conn_im_pyr[s, t, i] = connectivity_data[src][target][rec]
                    

    xticklabels_inh = []

    for t, target in enumerate(targets):
        for i, rec in enumerate(I_receptors):
              xticklabels_inh.append(f'{target}_{rec}')

    for s, src in enumerate(src_inh):
        for t, target in enumerate(targets):
            
            check_t = connectivity_data[src].get(target)
            if check_t is not None:
                for i, rec in enumerate(I_receptors):
                    if rec in connectivity_data[src][target]:
                        conn_im_inh[s, t, i] = -connectivity_data[src][target][rec]

    fig = plt.figure(figsize=(12, 4.), constrained_layout=True)
    gs = GridSpec(2, 3, width_ratios=[1, 0.025, 0.75], figure=fig)

    ax = [None]*3
    ax[0] = fig.add_subplot(gs[0, 0])
    ax[1] = fig.add_subplot(gs[1, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[1, 1])

    ax[2] = fig.add_subplot(gs[:, 2])
    #import ipdb; ipdb.set_trace()

    conn_im_inh = conn_im_inh.reshape(len(src_inh), len(targets)*2)

    conn_im_pyr = conn_im_pyr.reshape(len(src_pyr), len(targets)*2)

    norm_inh = Normalize(vmin=np.min(conn_im_inh), vmax=0)
    norm_pyr = Normalize(vmin=0, vmax=np.max(conn_im_pyr))

    im = ax[0].pcolormesh(conn_im_pyr, cmap='Reds', norm=norm_pyr)
    ax[0].set_yticks(np.arange(.5, 2.5, 1))
    ax[0].set_yticklabels(src_pyr, rotation='vertical', va='center')
    ax[0].set_xticks(np.arange(0.5, 8.5,1))
    ax[0].set_xticklabels(xticklabels_pyr)
    axxtw = ax[0].secondary_xaxis('top')
    axxtw.set_xticks(np.arange(0.5, 8.5,1))
    axxtw.set_xticklabels(xticklabels_pyr)
    #import ipdb; ipdb.set_trace()

    cb2 = plt.colorbar(im, cax=cax1)
    cb2.set_ticks([0, np.max(conn_im_pyr)])
    cb2.set_ticklabels(['0', f'{np.max(conn_im_pyr):.4f}'])
    cb2.set_label('synaptic weight ($mu S$)')


    im = ax[1].pcolormesh(conn_im_inh, cmap='Blues_r', norm=norm_inh)
    ax[1].set_yticks(np.arange(.5, 2.5, 1))
    ax[1].set_yticklabels(src_inh, rotation='vertical', va='center')
    ax[1].set_xticks(np.arange(0.5, 8.5,1))
    ax[1].set_xticklabels(xticklabels_inh)
    axxtw2 = ax[1].secondary_xaxis('top')
    axxtw2.set_xticks(np.arange(0.5, 8.5,1))
    axxtw2.set_xticklabels([])
    #import ipdb; ipdb.set_trace()

    cb1 = plt.colorbar(im, cax=cax2)
    cb1.set_ticks([np.min(conn_im_inh), 0])
    cb1.set_ticklabels([f'{np.min(conn_im_inh):.4f}', '0'])
    cb1.set_label('synaptic weight ($mu S$)')


    # space constant
    x = np.linspace(0, 300, 1000)
    y = conn_space_constant(x, lamtha1*30)
    y2 = conn_space_constant(x, lamtha2*30)
    ax[2].plot(x, 0.1*np.exp(-x**2/(2*129.66**2)), label='gauss with sigma=129.66, Campagnola, 2022', color='k')
    ax[2].plot(x, y, linestyle='--', label='conn with space constant lamtha=6.125*d', color='#88e788')
    ax[2].plot(x, y2, label='space constant lamtha=6.125*d*0.8', color='#198450')
    ax[2].legend()
    ax[2].set_xlabel('Distance (um)')
    ax[2].set_ylabel('Connection weight')
    ax[2].set_xlim([0, 300])
    ax[2].set_ylim([0, 0.11])
    ax[2].set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    ax[2].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])

    return fig

def plot_connectivity_space_constant_jones(net):

    targets = ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal'  ]
    src_inh = ['L2_basket', 'L5_basket']
    src_pyr = ['L2_pyramidal', 'L5_pyramidal']
    E_receptors = ['ampa', 'nmda']
    I_receptors = ['gabaa', 'gabab']

    connectivity_data = dict()

    for src in targets:
        connectivity_data[src] = dict()
        
        for conn in net.connectivity:
            if conn['src_type'] == src:
                t_type = conn['target_type']
                check_t = connectivity_data[src].get(t_type)
                if check_t is None:
                    connectivity_data[src][t_type] = dict()       
                rec = conn['receptor']
                connectivity_data[src][t_type][rec] = conn['nc_dict']['A_weight']


    conn_im_pyr = np.zeros((len(src_pyr), len(targets), 2))
    conn_im_inh = np.zeros((len(src_inh), len(targets), 2))

    xticklabels_pyr = []

    for t, target in enumerate(targets):
        for i, rec in enumerate(E_receptors):
              xticklabels_pyr.append(f'{target}_{rec}')

    for s, src in enumerate(src_pyr):
        for t, target in enumerate(targets):   
            check_t = connectivity_data[src].get(target)
            
            if check_t is not None:
                for i, rec in enumerate(E_receptors):
                    if rec in connectivity_data[src][target]:
                        conn_im_pyr[s, t, i] = connectivity_data[src][target][rec]
                    

    xticklabels_inh = []

    for t, target in enumerate(targets):
        for i, rec in enumerate(I_receptors):
              xticklabels_inh.append(f'{target}_{rec}')

    for s, src in enumerate(src_inh):
        for t, target in enumerate(targets):
            
            check_t = connectivity_data[src].get(target)
            if check_t is not None:
                for i, rec in enumerate(I_receptors):
                    if rec in connectivity_data[src][target]:
                        conn_im_inh[s, t, i] = -connectivity_data[src][target][rec]

    fig = plt.figure(figsize=(12, 4.), constrained_layout=True)
    gs = GridSpec(2, 3, width_ratios=[1, 0.025, 0.75], figure=fig)

    ax = [None]*3
    ax[0] = fig.add_subplot(gs[0, 0])
    ax[1] = fig.add_subplot(gs[1, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[1, 1])

    ax[2] = fig.add_subplot(gs[:, 2])
    #import ipdb; ipdb.set_trace()

    conn_im_inh = conn_im_inh.reshape(len(src_inh), len(targets)*2)

    conn_im_pyr = conn_im_pyr.reshape(len(src_pyr), len(targets)*2)

    norm_inh = Normalize(vmin=np.min(conn_im_inh), vmax=0)
    norm_pyr = Normalize(vmin=0, vmax=np.max(conn_im_pyr))

    im = ax[0].pcolormesh(conn_im_pyr, cmap='Reds', norm=norm_pyr)
    ax[0].set_yticks(np.arange(.5, 2.5, 1))
    ax[0].set_yticklabels(src_pyr, rotation='vertical', va='center')
    ax[0].set_xticks(np.arange(0.5, 8.5,1))
    ax[0].set_xticklabels(xticklabels_pyr)
    axxtw = ax[0].secondary_xaxis('top')
    axxtw.set_xticks(np.arange(0.5, 8.5,1))
    axxtw.set_xticklabels(xticklabels_pyr)
    #import ipdb; ipdb.set_trace()

    cb2 = plt.colorbar(im, cax=cax1)
    cb2.set_ticks([0, np.max(conn_im_pyr)])
    cb2.set_ticklabels(['0', f'{np.max(conn_im_pyr):.4f}'])
    cb2.set_label('synaptic weight ($mu S$)')


    im = ax[1].pcolormesh(conn_im_inh, cmap='Blues_r', norm=norm_inh)
    ax[1].set_yticks(np.arange(.5, 2.5, 1))
    ax[1].set_yticklabels(src_inh, rotation='vertical', va='center')
    ax[1].set_xticks(np.arange(0.5, 8.5,1))
    ax[1].set_xticklabels(xticklabels_inh)
    axxtw2 = ax[1].secondary_xaxis('top')
    axxtw2.set_xticks(np.arange(0.5, 8.5,1))
    axxtw2.set_xticklabels([])
    #import ipdb; ipdb.set_trace()

    cb1 = plt.colorbar(im, cax=cax2)
    cb1.set_ticks([np.min(conn_im_inh), 0])
    cb1.set_ticklabels([f'{np.min(conn_im_inh):.4f}', '0'])
    cb1.set_label('synaptic weight ($mu S$)')
    # space constant
    x = np.linspace(0, 300, 1000)
    y = conn_space_constant(x, 6.125*30)
    y1 = conn_space_constant(x, 3*30)
    y2 = conn_space_constant(x, 50*30)
    y3 = conn_space_constant(x, 20*30)
    ax[2].plot(x, y, label='Campagnola', color='k')
    ax[2].plot(x, y1, label='E-E', color='#2eb774')
    ax[2].plot(x, y2, label='I-E', color='#88e788')
    ax[2].plot(x, y3, label='I-I', color='#198450')
    ax[2].legend()
    ax[2].set_xlabel('Distance (um)')
    ax[2].set_ylabel('Connection weight')
    ax[2].set_xlim([0, 300])
    ax[2].set_ylim([0, 0.11])

    return fig

def plot_conn_heatmap(net):

    fig, ax = plt.subplots(3,3,figsize=(10, 10), sharex=True, sharey=True)

    # E-E

    conn = net.connectivity[2]

    pos_l5 = np.zeros((100, 3))
    for p, pos in enumerate(net.pos_dict['L5_pyramidal']):
        pos_l5[p] = np.array(pos)
        
    l5_gid = 170
    pos_dict_idx = int(l5_gid)-170
    ref_pos = net.pos_dict['L5_pyramidal'][pos_dict_idx]
    
    l5_center_conn_bl = np.zeros((10,10))

    for target_gid in conn["gid_pairs"][l5_gid]:
        if target_gid in net.gid_ranges['L5_pyramidal']:
            target_pos = net.pos_dict['L5_pyramidal'][int(target_gid-170)]
            weight, _ = _get_gaussian_connection(
                        ref_pos, target_pos, conn['nc_dict'],
                        inplane_distance=net._inplane_distance)

            x_idx = int(target_pos[0])
            y_idx = int(target_pos[1])
            l5_center_conn_bl[x_idx, y_idx] = weight

    ax[0,0].imshow(l5_center_conn_bl, origin='lower', cmap='hot', vmin=0, vmax=np.max(l5_center_conn_bl))
    ax[0,0].add_patch(patches.Rectangle((ref_pos[0]-0.5, ref_pos[1]-0.5), 1, 1, fill=False, edgecolor='violet', linewidth=2))

    l5_gid = 170 + 44
    pos_dict_idx = int(l5_gid)-170
    ref_pos = net.pos_dict['L5_pyramidal'][pos_dict_idx]
    l5_center_conn_c = np.zeros((10,10))
    for target_gid in conn["gid_pairs"][int(l5_gid)]:
        if target_gid in net.gid_ranges['L5_pyramidal']:
            target_pos = net.pos_dict['L5_pyramidal'][int(target_gid-170)]
            weight, _ = _get_gaussian_connection(
                        ref_pos, target_pos, conn['nc_dict'],
                        inplane_distance=net._inplane_distance)

            x_idx = int(target_pos[0])
            y_idx = int(target_pos[1])
            l5_center_conn_c[x_idx, y_idx] = weight
    ax[0,1].imshow(l5_center_conn_c, origin='lower', cmap='hot', vmin=0, vmax=np.max(l5_center_conn_c))
    ax[0,1].add_patch(patches.Rectangle((ref_pos[0]-0.5, ref_pos[1]-0.5), 1, 1, fill=False, edgecolor='violet', linewidth=2))

    l5_gid = 269
    pos_dict_idx = int(l5_gid)-170
    ref_pos = net.pos_dict['L5_pyramidal'][pos_dict_idx]
    l5_center_conn_tr = np.zeros((10,10))
    for target_gid in conn["gid_pairs"][l5_gid]:
        if target_gid in net.gid_ranges['L5_pyramidal']:
            target_pos = net.pos_dict['L5_pyramidal'][int(target_gid-170)]
            weight, _ = _get_gaussian_connection(
                        ref_pos, target_pos, conn['nc_dict'],
                        inplane_distance=net._inplane_distance)

            x_idx = int(target_pos[0])
            y_idx = int(target_pos[1])
            l5_center_conn_tr[x_idx, y_idx] = weight

    ax[0,2].imshow(l5_center_conn_tr, origin='lower', cmap='hot', vmin=0, vmax=np.max(l5_center_conn_tr))
    ax[0,2].add_patch(patches.Rectangle((ref_pos[0]-0.5, ref_pos[1]-0.5), 1, 1, fill=False, edgecolor='violet', linewidth=2))

    # E-I
    conn = net.connectivity[-2]
    l5_gid = 170
    pos_dict_idx = int(l5_gid)-170
    ref_pos = net.pos_dict['L5_pyramidal'][pos_dict_idx]
    l5_center_conn_bl_ei = np.zeros((10,10))
    for target_gid in conn["gid_pairs"][l5_gid]:
        if target_gid in net.gid_ranges['L5_basket']:
            target_pos = net.pos_dict['L5_basket'][int(target_gid-135)]
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        l5_center_conn_bl_ei[x_idx, y_idx] = weight
    ax[1,0].imshow(l5_center_conn_bl_ei, origin='lower', cmap='hot', vmin=0, vmax=np.max(l5_center_conn_bl_ei))
    ax[1,0].add_patch(patches.Rectangle((ref_pos[0]-0.5, ref_pos[1]-0.5), 1, 1, fill=False, edgecolor='violet', linewidth=2))

    l5_gid = 170 + 44
    pos_dict_idx = int(l5_gid)-170
    ref_pos = net.pos_dict['L5_pyramidal'][pos_dict_idx]
    l5_center_conn_c_ei = np.zeros((10,10))
    for target_gid in conn["gid_pairs"][l5_gid]:
        if target_gid in net.gid_ranges['L5_basket']:
            target_pos = net.pos_dict['L5_basket'][int(target_gid-135)]
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        l5_center_conn_c_ei[x_idx, y_idx] = weight
    ax[1,1].imshow(l5_center_conn_c_ei, origin='lower', cmap='hot', vmin=0, vmax=np.max(l5_center_conn_c_ei))
    ax[1,1].add_patch(patches.Rectangle((ref_pos[0]-0.5, ref_pos[1]-0.5), 1, 1, fill=False, edgecolor='violet', linewidth=2))

    l5_gid = 269
    pos_dict_idx = int(l5_gid)-170
    ref_pos = net.pos_dict['L5_pyramidal'][pos_dict_idx]
    l5_center_conn_tr_ei = np.zeros((10,10))
    for target_gid in conn["gid_pairs"][l5_gid]:
        if target_gid in net.gid_ranges['L5_basket']:
            target_pos = net.pos_dict['L5_basket'][int(target_gid-135)]
        weight, _ = _get_gaussian_connection(
                    ref_pos, target_pos, conn['nc_dict'],
                    inplane_distance=net._inplane_distance)

        x_idx = int(target_pos[0])
        y_idx = int(target_pos[1])
        l5_center_conn_tr_ei[x_idx, y_idx] = weight
    ax[1,2].imshow(l5_center_conn_tr_ei, origin='lower', cmap='hot', vmin=0, vmax=np.max(l5_center_conn_tr_ei))
    ax[1,2].add_patch(patches.Rectangle((ref_pos[0]-0.5, ref_pos[1]-0.5), 1, 1, fill=False, edgecolor='violet', linewidth=2)) 

    
    # I-I
    conn = net.connectivity[16]
    # import pdb
    # pdb.set_trace()
    pos_l5 = np.zeros((100, 3))
    for p, pos in enumerate(net.pos_dict['L2_basket']):
        pos_l5[p] = np.array(pos)
        
    inh_gid = 0
    pos_dict_idx = int(inh_gid)
    ref_pos = net.pos_dict['L2_basket'][pos_dict_idx]
    
    inh_center_conn_bl = np.zeros((10,10))
    for target_gid in conn["gid_pairs"][inh_gid]:
        if target_gid in net.gid_ranges['L2_basket']:
            target_pos = net.pos_dict['L2_basket'][int(target_gid)]
            weight, _ = _get_gaussian_connection(
                        ref_pos, target_pos, conn['nc_dict'],
                        inplane_distance=net._inplane_distance)

            x_idx = int(target_pos[0])
            y_idx = int(target_pos[1])
            inh_center_conn_bl[x_idx, y_idx] = weight
    ax[2,0].imshow(inh_center_conn_bl, origin='lower', cmap='hot')
    ax[2,0].add_patch(patches.Rectangle((ref_pos[1]-0.5, ref_pos[0]-0.5), 1, 1, fill=False, edgecolor='violet', linewidth=2))

    inh_gid = 15
    pos_dict_idx = int(inh_gid)
    ref_pos = net.pos_dict['L2_basket'][pos_dict_idx]
    inh_center_conn_c = np.zeros((10,10))
    for target_gid in conn["gid_pairs"][inh_gid]:
        if target_gid in net.gid_ranges['L2_basket']:
            target_pos = net.pos_dict['L2_basket'][int(target_gid)]
            weight, _ = _get_gaussian_connection(
                        ref_pos, target_pos, conn['nc_dict'],
                        inplane_distance=net._inplane_distance)

            x_idx = int(target_pos[0])
            y_idx = int(target_pos[1])
            inh_center_conn_c[x_idx, y_idx] = weight
    ax[2,1].imshow(inh_center_conn_c, origin='lower', cmap='hot')
    ax[2,1].add_patch(patches.Rectangle((ref_pos[1]-0.5, ref_pos[0]-0.5), 1, 1, fill=False, edgecolor='violet', linewidth=2))

    inh_gid = 34
    pos_dict_idx = int(inh_gid)
    ref_pos = net.pos_dict['L2_basket'][pos_dict_idx]
    inh_center_conn_tr = np.zeros((10,10))
    for target_gid in conn["gid_pairs"][inh_gid]:
        if target_gid in net.gid_ranges['L2_basket']:
            target_pos = net.pos_dict['L2_basket'][int(target_gid)]
            weight, _ = _get_gaussian_connection(
                        ref_pos, target_pos, conn['nc_dict'],
                        inplane_distance=net._inplane_distance)

            x_idx = int(target_pos[0])
            y_idx = int(target_pos[1])
            inh_center_conn_tr[x_idx, y_idx] = weight

    im = ax[2,2].imshow(inh_center_conn_tr, origin='lower', cmap='hot')
    ax[2,2].add_patch(patches.Rectangle((ref_pos[1]-0.5, ref_pos[0]-0.5), 1, 1, fill=False, edgecolor='violet', linewidth=2))

    cb = fig.colorbar(im, ax=ax, shrink=0.1, fraction=0.1, aspect=7, label='syn. weight', orientation='horizontal')
    cb.set_ticks([0, inh_center_conn_tr.max()])
    cb.set_ticklabels([f'0', 'max'])

    for a in ax.flatten():
        a.set_xlim([-.5, 9.5])
        a.set_ylim([-.5, 9.5])
        a.set_xticks([0, 9])
        a.set_yticks([0, 9])
        a.set_xticklabels([1, 10])
        a.set_yticklabels([1, 10])

    return fig


def plot_interlayer_conn():

    fig, ax = plt.subplots(2, 4, figsize=(20, 4), height_ratios=[0.05, 1], sharex=True)

    cell_types = ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal']
    for i,c in enumerate(cell_types):
        # stimulate L2/3 pyramidal
        net = duecker_ET_model()

        net.set_cell_positions(inplane_distance=30.)

        ampl = 3.
        t0 = 10

        net.add_tonic_bias(cell_type=c,section='soma',bias_name='tonic_soma',amplitude=ampl,t0=t0,tstop=t0+3)

        #with MPIBackend(n_procs=10, mpi_cmd='mpiexec'):
        dpl = simulate_dipole(net, tstop=30, n_trials=1)

        # clamp signal for plotting
        timevec = net.cell_response.times

        clamp = np.zeros_like(timevec) 

        for bias in net.external_biases:
            for cell_type in net.external_biases[bias]:
                t0 = np.argmin(np.abs(timevec - net.external_biases[bias][cell_type]['t0']))
                tstop = np.argmin(np.abs(timevec - net.external_biases[bias][cell_type]['tstop']))

                clamp[t0:tstop] = net.external_biases[bias][cell_type]['amplitude']
        
        ax[0, i].plot(timevec, clamp, color='k', linewidth=1.5)
        ax[0, i].set_ylabel('clamp (nA)')
        ax[0, i].set_title(f'{cell_types[i]}')
        ax[0, i].set_xlim([5, 25])
        ax[0, i].set_axis_off()

        I = np.array(net.cell_response.spike_gids[0]) <270
        gids = np.array(net.cell_response.spike_gids).reshape(-1,)[I]
        times = np.array(net.cell_response.spike_times).reshape(-1,)[I]
        cell_types = ['L2_inhibitory', 'L2_pyramidal', 'L5_inhibitory', 'L5_pyramidal']
        cell_type_colors = {'L2_basket': cmap.colors[0], 'L2_pyramidal': cmap.colors[1], 
                    'L5_basket': cmap.colors[2], 'L5_pyramidal': cmap.colors[3]}

        colors = [assign_color(net, gid) for gid in gids]

        ax[1, i].scatter(times, gids, s=8, alpha=1., c=colors, marker='|')
        ax[1, i].set_ylabel('neuron index')
        ax[1, i].set_xlim([5, 25])
        ax[1, i].set_xticks(np.arange(5, 26, 10))
        ax[1, i].set_xlabel('time (ms)')
        ax[1, i].set_ylim([0, 270])
        ax[1, i].set_yticks([0, 35, 135, 170, 270])
        ax[1, i].invert_yaxis()

    fig.tight_layout()
    return fig
    
def plot_psps():

    net_no_stim = duecker_ET_model()
    dpl = simulate_dipole(
        net_no_stim, tstop=40, dt=0.05, bsl_cor="duecker", record_vsec="soma"
    )

    def get_psp(src_cell, trgt_cell, ampl,tsamp,):

        net = duecker_ET_model()
        g = net_no_stim.gid_ranges[src_cell][int(len(net_no_stim.gid_ranges[src_cell])/2)]
        psp_gid = np.setdiff1d(net.gid_ranges[trgt_cell], g)

        num_sweeps = len(net.gid_ranges[trgt_cell])
        if src_cell == trgt_cell:
            num_sweeps -=1
        psp = np.zeros((num_sweeps,1))
        v_mem = np.zeros((num_sweeps, tsamp))

        net.add_tonic_bias(cell_type=src_cell, gid=g, bias_name="tonic_soma", amplitude=ampl, t0=10, tstop=15)

        dpl = simulate_dipole(net, tstop=40, dt=0.05, bsl_cor="duecker", record_vsec="soma")

        for ei, e in enumerate(psp_gid):
            v = np.array(net.cell_response.vsec[0][e]["soma"])
            bsl = np.array(net_no_stim.cell_response.vsec[0][e]["soma"])
            psp[ei] = np.max(np.abs(v - bsl))
            v_mem[ei] = v

        return psp, v_mem
    
    fig, ax = plt.subplots(5, 4,figsize=(6, 8), constrained_layout=True, width_ratios=[1, 0.5, 1, 0.5])
    ax = ax.flatten()
    src_cells = ["L2_pyramidal", "L5_pyramidal", "L2_pyramidal", "L2_pyramidal", "L2_pyramidal", "L5_pyramidal", "L2_basket", "L5_basket", "L2_basket", "L5_basket"]
    trgt_cells = ["L2_pyramidal", "L5_pyramidal", "L5_pyramidal", "L2_basket", "L5_basket", "L5_basket", "L2_pyramidal", "L5_pyramidal", "L5_pyramidal", "L5_basket"]
    neuron_cols = {"L2_basket": '#d4a498',
                "L2_pyramidal": '#941a49',
                "L5_basket": '#6e97ae',
                "L5_pyramidal": '#3a5a8a'}

    aidx = 0
    for src, trgt in zip(src_cells, trgt_cells):

        if src == "L5_pyramidal":
            ampl = 3
        elif src == "L2_pyramidal":
            ampl = 1
        elif src == "L2_basket":
            ampl = 0.5

        psp, vmem = get_psp(src_cell=src, trgt_cell=trgt, ampl=ampl, tsamp=len(dpl[0].times))
        ax[aidx].plot(dpl[0].times, vmem.T, color=neuron_cols[trgt], alpha=0.3)
        ax[aidx].plot(dpl[0].times, vmem.mean(axis=0), color='k', linewidth=3)
        ax[aidx].spines['top'].set_visible(False)
        ax[aidx].spines['right'].set_visible(False)
        ax[aidx].set_xlim(left=10)
        ax[aidx].set_title('')
        
        ax[aidx].set_ylim([np.round(vmem.min()-0.1,1), np.round(vmem.max()+0.1,1)])
        ax[aidx].set_yticks([np.round(vmem.min(),1)-0.1, np.round(vmem.max()+0.1,1)])
        ax[aidx].set_xlim([10, 40])
        ax[aidx].set_xticks([10, 40])
        ax[aidx].set_title(f'{src} - {trgt}')
        aidx +=1
        
        if src == 'L5_pyramidal' or src == 'L2_pyramidal':
            pmin = 0
            pmax = np.round(psp.max()+0.1,1)
        else:
            psp *= -1
            pmin = np.round(psp.min()-0.01,1)
            pmax = 0
        v = ax[aidx].violinplot(psp, showmedians=True)
        v['bodies'][0].set_alpha(0.3)
        v['bodies'][0].set_facecolor(neuron_cols[trgt])
        v['cbars'].set_color(neuron_cols[trgt])
        v['cmaxes'].set_color(neuron_cols[trgt])
        v['cmins'].set_color(neuron_cols[trgt])
        v['cmedians'].set_color(neuron_cols[trgt])
        ax[aidx].scatter(np.random.uniform(0.95,1.05,len(psp)), psp, s=10, alpha=0.75, c=neuron_cols[trgt])
        ax[aidx].text(1, np.median(psp), np.round(np.median(psp),2))
        ax[aidx].spines['top'].set_visible(False)
        ax[aidx].spines['right'].set_visible(False)
        ax[aidx].spines['bottom'].set_visible(False)
        ax[aidx].set_xticks([])
        
        ax[aidx].set_ylim([pmin, pmax])
        ax[aidx].set_yticks([pmin, pmax])
        aidx+=1

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
            l5_name = "L5_pyramidal"
        else:
            net = duecker_ET_model(mesh_shape=(1,1))
            l5_name = "L5_pyramidal"

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
                rheo[i] = rheo[0]
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
        ax[0, i].plot(jonnet.cell_response.times, jonnet.cell_response.vsec[0][c]['soma'], color='k')
        ax[1, i].plot(jonnet.cell_response.times, canet.cell_response.vsec[0][c]['soma'], color='k')
        ax[0, i].set_title(f'{cell_types[c]}: {rheo_jones[c]:.4f} nA')
        ax[1, i].set_title(f'{cell_types[c]}: {rheo_ca[c]:.4f} nA')
        ax[1, i].set_xlabel('time (ms)')
        ax[0, 0].set_ylabel('membrane potential ($\mu$V)')
        ax[0, i].set_ylim(-100, 50)
        ax[0, i].set_ylim(-100, 50)
        ax[0, i].set_xlim(0, 100)
        ax[0, i].set_xticks([0, 50, 100])

        ax[0, i].spines['top'].set_visible(False)
        ax[0, i].spines['right'].set_visible(False)
        ax[1, i].spines['top'].set_visible(False)
        ax[1, i].spines['right'].set_visible(False)
        ax[0, i].spines['bottom'].set_bounds(0, 50)
        ax[1, i].spines['bottom'].set_bounds(0, 50)
        ax[0, i].spines['left'].set_bounds(0, 50)
        ax[1, i].spines['left'].set_bounds(0, 50)
        ax[0, i].set_yticks([50])
        ax[0, i].set_xticks([50])
       
    fig.tight_layout()
    h('forall delete_section()')

    return fig


