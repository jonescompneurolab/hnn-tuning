"""Single-cell pyramidal stimulation experiments and plotting.

Shared by the L2/3 and L5-ET single-cell notebooks.
Importing this module loads the custom ``epsp.mod`` point process (needed for the
attenuation experiment).

Author: Katharina Duecker katharina.duecker@gmail.com
Last checked: 18-Oct-2026
"""
import neuron
from neuron import h
from neuron.units import ms
import numpy as np
from scipy import signal

from hnn_core import simulate_dipole

from ._paths import EPSP_MOD
from .plot_utils import add_scalebar

# directory holding the custom epsp.mod point process
neuron.load_mechanisms(str(EPSP_MOD))


# Direct-NEURON single-cell experiments (attenuation along the apical dendrite)
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
    ax.set_ylabel('membrane potential (mV)')
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

            v = dend_voltg[i].as_numpy()
            d_filt = v 
            r_filt = ref_voltg.as_numpy() 


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


    atten = np.array(amp_dend)/np.array(amp_ref)*100

    idx = np.argsort(distance)
    atten = atten[idx]
    distance = np.array(distance)[idx]

    if amp>0:
        upstroke = np.array(upstroke)[idx]

    return atten, distance, upstroke, ax


# Single cell dipole (hnn-core)
def stim_dpl(net, stim_cell='L5_pyramidal', stim_section='soma', stim_length=20, ampl=2.5):
    """Drive one isolated cell with a tonic bias and return its dipole.

    Parameters
    ----------
    net : hnn_core.Network
        A network (e.g. ``duecker_ET_model(mesh_shape=(1, 1))``); connectivity is
        cleared so the stimulated cell is isolated.
    stim_cell : str
        Cell type to stimulate (``'L2_pyramidal'`` or ``'L5_pyramidal'``).
    stim_section : str
        Section receiving the tonic bias (e.g. ``'soma'`` or ``'apical_tuft'``).
    stim_length : float
        Duration of the tonic bias in ms.
    ampl : float
        Bias amplitude in nA.

    Returns
    -------
    net, dpl, smooth_dpl
        The simulated network, the raw dipole, and a smoothed copy.
    """
    window_len, scaling_factor = 10, 1

    net.clear_connectivity()

    net.set_cell_positions(inplane_distance=30.)

    depths = list(range(-100, 2400, 200))
    electrode_pos = [(135, 135, dep) for dep in depths]

    net.add_electrode_array('probe1', electrode_pos)

    t0 = 20

    net.add_tonic_bias(cell_type=stim_cell, section=stim_section, bias_name='pulse',
                       amplitude=ampl, t0=t0, tstop=t0 + stim_length)

    dpl = simulate_dipole(net, tstop=80, n_trials=1, record_vsec='all',
                          record_isec='all', record_ca='all')
    smooth_dpl = dpl[0].copy()
    smooth_dpl.smooth(window_len).scale(scaling_factor)

    return net, dpl, smooth_dpl


def plot_dpl(net, dpl, smooth_dpl, axs, gid, cell_type,
             cells=['L5_pyramidal', 'L2_pyramidal'], stim_length=20, stim='clamp',
             sb_time=25, sb_mV=50, sb_dpl=None, dpl_ylim=(-0.3, 0.3)):
    """Plot soma/apical-tuft membrane potential and dipole moment for one cell.

    ``axs`` is a length-2 sequence of axes: the first holds the membrane
    potential traces, the second the raw and smoothed dipole moment (nAm/mm^2).
    ``sb_time``/``sb_mV``/``sb_dpl`` are the scale-bar lengths (ms / mV / dipole)
    for the two panels; override per cell type as needed.
    """
    cols = np.array(([67, 104, 158], [164, 30, 79], [117, 112, 179], [231, 41, 138], [102, 166, 30])) / 255
    c = cells.index(cell_type)

    timevec = net.cell_response.times

    v = net.cell_response.vsec[0][gid]['soma']
    v_apcl = net.cell_response.vsec[0][gid]['apical_tuft']

    if cell_type == 'L5_pyramidal':
        alpha = 1
        color_soma = 'k'
    else:
        alpha = 0.5
        color_soma = cols[c]

    axs[0].plot(timevec, v, color=color_soma, linewidth=1.5, alpha=alpha, label='soma')
    axs[0].plot(timevec, v_apcl, color=cols[c], linewidth=1.5, label='apical tuft')
    axs[0].legend(loc='lower right', fontsize=8)
    axs[0].set_ylim([-90, 60])

    if stim == 'clamp':
        clamp = np.zeros_like(v)
        t0 = np.argmin(np.abs(timevec - 20))
        t1 = np.argmin(np.abs(timevec - (20 + stim_length)))
        clamp[t0:t1] = 2.5
        axs[0].plot(timevec, clamp - 85, color='r', linewidth=1., label='clamp')

    # convert dipole to nAm/mm2
    net_dict = net.cell_types[cell_type]['cell_object'].to_dict()
    basal_plane = np.abs(net_dict['sections']['basal_2']['end_pts'][1][0] - net_dict['sections']['basal_3']['end_pts'][1][0]) * net_dict['sections']['soma']['diam']

    surface_mm2 = basal_plane * 1e-6

    dpl_mom = dpl[0].data[cell_type[:2]] / len(net.gid_ranges[cell_type]) / surface_mm2
    max_mom = np.abs(dpl_mom).max()
    axs[1].plot(timevec, dpl_mom, label='unsmoothed', color='k', linewidth=2, alpha=0.5)
    axs[1].text(25, np.round(max_mom, 2), str(np.round(max_mom, 2)))
    axs[1].plot(timevec, smooth_dpl.data[cell_type[:2]] / len(net.gid_ranges[cell_type]) / surface_mm2, label='smoothed', color='k', linewidth=2)
    axs[1].legend(loc='lower right', fontsize=8)
    axs[1].set_title('')
    

    # dipole y-range: fixed (-0.3, 0.3) suits L2/3; pass dpl_ylim=None to autoscale
    # to the data (needed for L5, whose dipole is much larger).
    if dpl_ylim is None:
        lim = np.round(max_mom * 1.2, 2)
        lim = lim if lim > 0 else 0.1
        dpl_ylim = (-lim, lim)
    axs[1].set_ylim(dpl_ylim)
    # size the dipole scale bar to the panel unless the caller fixed it
    if sb_dpl is None:
        sb_dpl = np.round((dpl_ylim[1] - dpl_ylim[0]) / 3, 2)

    # scale bars (replace per-panel spine.set_bounds; aligned via axes-fraction anchor)
    add_scalebar(axs[0], x_len=sb_time, y_len=sb_mV,
                 x_label=f'{sb_time:g} ms', y_label=f'{sb_mV:g} mV')
    add_scalebar(axs[1], x_len=sb_time, y_len=sb_dpl,
                 x_label=f'{sb_time:g} ms', y_label=f'{sb_dpl:g} nAm/mm2')

    return axs
