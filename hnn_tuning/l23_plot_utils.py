"""L2/3 plot utils """

import neuron
from neuron import h
import matplotlib.pyplot as plt
import matplotlib
from neuron.units import ms, mV
import numpy as np
import os
from ._paths import HNN_CORE_MOD, results_dir

pth_mod = str(HNN_CORE_MOD)
figpth = str(results_dir('single_cell_sim', 'l23'))
plt.rcParams['svg.fonttype'] = 'none'

from .cell_builder_utils import l2_params, _get_dends, _get_pyr_soma, Cell
from .l23_mechs_utils import Mechs
from functools import partial
from .l5_mechs_utils import linear_gbar, Ih_condu_hay
from hnn_core.viz import plot_cell_morphology
from matplotlib import colormaps
from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from .model_data_comparison import exp_func

def exp_decay(x, lambda_val, A):
    return A * np.exp(-x / lambda_val)

def get_colormap_l23(cmap_name='inferno'):

    cmap = colormaps['inferno']
    cols_dist = cmap.colors[0:len(cmap.colors):17]
    # neuron line colours now live in hnn_tuning.plot_utils.get_cmap_neurons
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

    return im, cols_dist, cols_apcl, cols_basal

def get_model_params():

    return {'soma': {'NaTs2_t_32d_hay2011': 20_400e-4*1.2,
                                'SKv3_1_hay2011': 6_930e-4*1.5,
                                'Nap_Et2_hay2011': 1e-4/2*0,
                                'Ca_HVA_hay2011': 9.92e-04,
                                'Ca_LVAst_hay2011': 34.3e-4,

                                'SK_E2_hay2011': 3.e-05,
                                'pas': 1.0 / 37000,
                                'Ih_hay2011': 0.000080/4,
                                'Im_hay2011': 8e-3*0.05,
                                'CaDynamics_E2_hay2011': {'decay': 90, 'gamma': 0.00053300}},
                        'basal': {'NaTs2_t_32d_hay2011': 0.008009,
                                  'SKv3_1_hay2011': 0.000513,
                                  'Ih_hay2011': 0.00008/4,
                                  'pas':  1.0 / 37000.0},
                        'dend': {'NaTa_t_32d_hay2011': 0.008*0.01,
                                'SKv3_1_hay2011': 0.003*4,
                                'Ca_HVA_hay2011': partial(linear_gbar, 0.00001, 0.002, 200, hotzone=[200, 400], hotzone_factor=4),
                                'Ca_LVAst_hay2011': partial(linear_gbar, 0.0000001, 0.001, 200, hotzone=[200, 400], hotzone_factor=4),
                                'pas': 1.0 /37000,
                                'SK_E2_hay2011': partial(linear_gbar, 3.e-06, 3.e-03, 200, hotzone=[200, 400], hotzone_factor=10),
                                'Ih_hay2011': partial(Ih_condu_hay, 0.00008/2),
                                'Im_hay2011': 0.0009*0.0001,
                                'CaDynamics_E2_hay2011': {'decay': 70, 'gamma': 0.00053300*2.5}},
                        'e_pas': -75}

def get_l2pyr(v_init={
        "soma": -73.50426114124605,
        "apical_trunk": -73.47809226638849,
        "apical_oblique": -73.47597057498221,
        "apical_1": -73.28712400156321,
        "apical_tuft": -73.0318451642059,
        "basal_1": -73.51253755658675,
        "basal_2": -73.53697581734804,
        "basal_3": -73.53697581734804,
    }):
                
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
            gbar_Ih_apical.append(seg.gbar_Ih_hay2011)
            gbar_Ca_HVA_apical.append(seg.gbar_Ca_HVA_hay2011)
            gbar_Ca_LVA_apical.append(seg.gbar_Ca_LVAst_hay2011)
            gbar_SK_E2_apical.append(seg.gbar_SK_E2_hay2011)

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

    fig, ax = plt.subplots(1,4,figsize=(16,4), constrained_layout=True)
    ax = ax.flatten()
    ax[0].plot(distance_apical, gbar_Ih_apical, color='k')
    ax[0].set_ylabel('$gbar_{Ih} (S/cm^2)$')
    ax[0].set_ylim([0, 0.0005])
    ax[0].set_yticks(np.arange(0, 0.00051, 0.00025))
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
    ax[1].set_xlabel('distance to soma ($\mu m$)')
    ax[0].set_xlabel('distance to soma ($\mu m$)')

    for a in ax:
        a.set_xlim([50, 600])
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    return fig
