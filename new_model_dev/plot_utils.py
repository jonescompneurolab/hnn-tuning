"""" Plotting functions for exploring Hay model and HNN """
from neuron import h
from neuron.units import ms, mV

import matplotlib.pyplot as plt
import numpy as np

def fig_conductances(model, apical_sections=['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft']):
    distance_apical = []
    gbar_Ih_apical = []
    gbar_Ca_HVA_apical = []
    gbar_Ca_LVA_apical = []

    h.distance(sec = model._nrn_sections['soma'])
    for section in apical_sections:
        for seg in model._nrn_sections[section]:
            distance_apical.append(h.distance(seg.x, sec=seg.sec))
            gbar_Ih_apical.append(seg.gbar_Ih)
            gbar_Ca_HVA_apical.append(seg.gbar_Ca_HVA)
            gbar_Ca_LVA_apical.append(seg.gbar_Ca_LVAst)

    fig, ax = plt.subplots(1,3,figsize=(12,3))
    ax[0].plot(distance_apical, gbar_Ih_apical)
    ax[0].set_ylabel('$gbar_{Ih} (S/cm^2)$')
    ax[0].set_xlabel('distance to soma ($\mu m$)')

    ax[1].plot(distance_apical, gbar_Ca_HVA_apical)
    ax[1].set_ylabel('$gbar_{Ca_{HVA}} (S/cm^2)$')
    ax[1].set_xlabel('distance to soma ($\mu m$)')


    ax[2].plot(distance_apical, gbar_Ca_LVA_apical)
    ax[2].set_ylabel('$gbar_{Ca_{LVA}} (S/cm^2)$')
    ax[2].set_xlabel('distance to soma ($\mu m$)')

    fig.tight_layout()

    return fig

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

    fig, ax = plt.subplots(2,2, figsize=(8,8), height_ratios=[0.25, 1], sharex=True)

    for c in clamp_current:
        ax[0,0].plot(t,c, color='k')
        ax[0,1].plot(t,c, color='k')

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
            ax[1,0].plot(t,np.ones_like(ca[i]) + cv, color='k', linestyle='-.')

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