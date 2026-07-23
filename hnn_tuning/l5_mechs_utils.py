"""mechs integrator utils"""
import neuron
from neuron import h
from functools import partial
import numpy as np

def Ih_condu_hay(gsoma, x):

    sl = 1.
    a = 2.087
    gbar = gsoma * (-0.8696 + a * np.exp(sl*(x/323)))

    return gbar

def linear_gbar(gsoma, gdend, xkink, x, hotzone=[0,0], hotzone_factor=1):

    gbar = gsoma + np.min([xkink, x]) * (gdend - gsoma) / xkink

    if x > hotzone[0] and x < hotzone[1]:
        gbar *= hotzone_factor

    return gbar

class Mechs:

    def __init__(self, path_mod):
        neuron.load_mechanisms(path_mod)
        h.load_file("stdrun.hoc")


    def add_mechs(self,cell, params):

        h.distance(sec = cell._nrn_sections['soma'])
        # apical_oblique as part of basal because it's the target of the proximal drive
        basal_sections = ['basal_1', 'basal_2', 'basal_3', 'apical_oblique']

        mechs = params['basal'].keys()
        for sec_name in basal_sections:
            for i, m in enumerate(mechs):
                cell._nrn_sections[sec_name].insert(m)
                for seg in cell._nrn_sections[sec_name]:
                    if m == 'pas':
                        seg.pas.g = params['basal']['pas']['g']
                        seg.pas.e = params['basal']['pas']['e']
                    else:
                        gbar = params['basal'][m]
                        if m == 'NaTs2_t_hay2011' or m == 'SKv3_1_hay2011':
                            distance = h.distance(seg.x, sec=seg.sec)
                            gbar = gbar(distance)
                        setattr(seg, 'gbar_'+m, gbar)
            cell._nrn_sections[sec_name].insert('CaDynamics_E2_hay2011')
            for seg in cell._nrn_sections[sec_name]:
                seg.CaDynamics_E2_hay2011.decay = 50
                seg.CaDynamics_E2_hay2011.gamma = .0005096
            
        # soma
        mechs = params['soma'].keys()
        for i,m in enumerate(mechs):
            cell._nrn_sections['soma'].insert(m)
            for seg in cell._nrn_sections['soma']:
                if m == 'pas':
                    seg.pas.g = params['soma']['pas']['g']
                    seg.pas.e = params['soma']['pas']['e']
                else:
                    setattr(seg, 'gbar_'+m, params['soma'][m])

        cell._nrn_sections['soma'].insert('CaDynamics_E2_hay2011')
        for seg in cell._nrn_sections['soma']:
            seg.CaDynamics_E2_hay2011.decay = 460
            seg.CaDynamics_E2_hay2011.gamma = .000501


        # apical sections
        apical_sections = ['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft']
        
        mechs = params['dend'].keys()
        for sec_name in apical_sections:
            for i,m in enumerate(mechs):
                cell._nrn_sections[sec_name].insert(m)
                for seg in cell._nrn_sections[sec_name]:
                    if m == 'pas':
                        distance = h.distance(seg.x, sec=seg.sec) 
                        seg.pas.g = params['dend']['pas']['g']             
                        seg.pas.e = params['dend']['pas']['e']    
                    elif hasattr(params['dend'][m], "__call__"):
                            distance = h.distance(seg.x, sec=seg.sec)
                            gbar = params['dend'][m](distance)
                            setattr(seg,'gbar_'+m, gbar)
                    else:
                        setattr(seg, 'gbar_'+m, params['dend'][m])       # Hay increases all conductances by 2 because cm is also increased by 2

            cell._nrn_sections[sec_name].insert('CaDynamics_E2_hay2011')
            for seg in cell._nrn_sections[sec_name]:
                seg.CaDynamics_E2_hay2011.decay = 122
                seg.CaDynamics_E2_hay2011.gamma = .0005096*1.


        return cell
    