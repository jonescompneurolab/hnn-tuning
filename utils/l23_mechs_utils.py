"""mechs integrator utils"""
import neuron
from neuron import h
from functools import partial
import numpy as np

# def Ih_condu_apical(gbar, offset, slope, factor_exp, x):

#     return gbar * (offset +  slope * np.exp(x*factor_exp))

# def distribute_gbar(gbar, slope, offset, x):

#     return (slope * x + offset) * gbar

def linear_gbar(gsoma, gdend, xkink, x, hotzone=[0,0], hotzone_factor=1):

    gbar = gsoma + np.min([xkink, x]) * (gdend - gsoma) / xkink

    if x > hotzone[0] and x < hotzone[1]:
        gbar *= hotzone_factor
    elif x > hotzone[1]:
        gbar = gsoma

    return gbar



class Mechs:

    def __init__(self, path_mod):
        neuron.load_mechanisms(path_mod)
        h.load_file("stdrun.hoc")


    def add_mechs(self,cell, params):

        h.distance(sec = cell._nrn_sections['soma'])
        #params = get_rich_params()
        # apical_oblique as part of basal because it's the target of the proximal drive?
        basal_sections = ['basal_1', 'basal_2', 'basal_3', 'apical_oblique']

        mechs = params['basal'].keys()
        for sec_name in basal_sections:
            cell._nrn_sections[sec_name].insert('extracellular')   # to set 0 extracellular potential
            for i,m in enumerate(mechs):
                cell._nrn_sections[sec_name].insert(m)
                for seg in cell._nrn_sections[sec_name]:
                    if m == 'pas':
                        seg.pas.g = params['basal']['pas']
                        seg.pas.e = params['e_pas']
                    else:
                        setattr(seg, 'gbar_'+m, params['basal'][m])

        # soma
        mechs = params['soma'].keys()
        cell._nrn_sections['soma'].insert('extracellular')   # to set 0 extracellular potential
        for i,m in enumerate(mechs):
            cell._nrn_sections['soma'].insert(m)
            for seg in cell._nrn_sections['soma']:
                if m == 'pas':
                    seg.pas.g = params['soma']['pas']
                    seg.pas.e = params['e_pas']
                elif m == 'CaDynamics_E2_hay2011':
                        cell._nrn_sections[sec_name].insert(m)     
                        seg.CaDynamics_E2_hay2011.decay = params['dend'][m]['decay']
                        seg.CaDynamics_E2_hay2011.gamma = params['dend'][m]['gamma']
                else:
                    setattr(seg, 'gbar_'+m, params['soma'][m])


        # apical sections
        apical_sections = ['apical_trunk', 'apical_1', 'apical_tuft']

        mechs = params['dend'].keys()
        h.distance(sec = cell._nrn_sections['soma'])
        for sec_name in apical_sections:
            cell._nrn_sections[sec_name].insert('extracellular')   # to set 0 extracellular potential
            for i,m in enumerate(mechs):
                cell._nrn_sections[sec_name].insert(m)
                for seg in cell._nrn_sections[sec_name]:
                    
                    distance = h.distance(seg.x, sec=seg.sec)
                    if m == 'pas':
                        if hasattr(params['dend'][m], "__call__"):
                            seg.pas.g = params['dend']['pas'](distance)              
                        else: 
                            seg.pas.g = params['dend']['pas']
                        seg.pas.e = params['e_pas']
                    elif m == 'CaDynamics_E2_hay2011':
                        cell._nrn_sections[sec_name].insert(m)     
                        seg.CaDynamics_E2_hay2011.decay = params['dend'][m]['decay']
                        seg.CaDynamics_E2_hay2011.gamma = params['dend'][m]['gamma']
                    else:
                        if hasattr(params['dend'][m], "__call__"):
                            gbar = params['dend'][m](distance)
                            setattr(seg,'gbar_'+m, gbar)
                        else:
                            setattr(seg,'gbar_'+m, params['dend'][m])

        return cell
    