"""mechs integrator utils"""
import neuron
from neuron import h
from functools import partial
import numpy as np

def Ih_condu_apical(gsoma, x):

    sl = 1.1
    a = 2.087-1
    return gsoma * (-0.8696 + a * np.exp(sl*(x/323)))

def Ih_condu_hay(gsoma, x):

    sl = 1.
    a = 2.087
    return gsoma * (-0.8696 + a * np.exp(sl*(x/323)))

def linear_gbar(gsoma, gdend, xkink, x):

    return gsoma + np.min([xkink, x]) * (gdend - gsoma) / xkink

def get_hay_params():

    hay_model_gbar = {'soma': {'NaTs2_t': 20_400e-4, 
                                'SKv3_1': 6_930e-4,
                                'Nap_Et2': 17.2e-4,
                                'Ca_HVA': 9.92e-04,
                                'Ca_LVAst': 34.3e-4,
                                'SK_E2': 441e-4,
                                'pas': .338e-4,
                                'Ih': 1e-4,
                                'Im': 0,
                                'K_Pst': 22.3e-4,
                                'K_Tst': 812e-4},
                        'dend': {'NaTa_t': 0.0213,
                                'SKv3_1': 0.000261,
                                'Nap_Et2': 0,
                                'Ca_HVA': 2.78e-5,
                                'Ca_LVAst': 93.5e-6,
                                'SK_E2': 0.0012,
                                'pas': 0.0000589,
                                'Ih': partial(Ih_condu_apical, 1e-4),
                                'Im': 0.0000675,
                                'K_Pst': 0,
                                'K_Tst': 0},
                        'e_pas': -90}
    
    return hay_model_gbar


class Mechs:

    def __init__(self, path_mod):
        neuron.load_mechanisms(path_mod)
        h.load_file("stdrun.hoc")


    def add_mechs(self,cell, params, hotzone=[650, 950]):

        h.distance(sec = cell._nrn_sections['soma'])
        #params = get_rich_params()
        # apical_oblique as part of basal because it's the target of the proximal drive?
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
                        if m == 'NaTs2_t' or m == 'SKv3_1':
                            distance = h.distance(seg.x, sec=seg.sec)
                            gbar = gbar(distance)
                        setattr(seg, 'gbar_'+m, gbar)
            
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

        cell._nrn_sections['soma'].insert('CaDynamics_E2')
        for seg in cell._nrn_sections['soma']:
            seg.CaDynamics_E2.decay = 460
            seg.CaDynamics_E2.gamma = .000501


        # apical sections
        apical_sections = ['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft']
        
        mechs = params['dend'].keys()
        for sec_name in apical_sections:
            for i,m in enumerate(mechs):
                cell._nrn_sections[sec_name].insert(m)
                for seg in cell._nrn_sections[sec_name]:
                    if m == 'pas':
                        seg.pas.g = params['dend']['pas']['g']             # Hay increases all conductances by 2 because resistance is also increased by 2
                        seg.pas.e = params['dend']['pas']['e']       
                    elif m == 'Ih':
                        distance = h.distance(seg.x, sec=seg.sec)
                        gbar = params['dend']['Ih'](distance)
                        setattr(seg, 'gbar_'+m, gbar)
                    
                    # set calcium according to function (new)
                    elif m == 'Ca_HVA':
                        distance = h.distance(seg.x, sec=seg.sec)
                        gbar = params['dend']['Ca_HVA'](distance)
                        if (distance >= hotzone[0]) and (distance < hotzone[1]):
                            gbar *= 3.9
                        setattr(seg, 'gbar_'+m, gbar)
                    elif m == 'Ca_LVAst':
                        distance = h.distance(seg.x, sec=seg.sec)
                        gbar = params['dend']['Ca_LVAst'](distance)
                        if (distance >= hotzone[0]) and (distance < hotzone[1]):
                            gbar *= 3
                        setattr(seg, 'gbar_'+m, gbar) 
                    else:
                        setattr(seg, 'gbar_'+m, params['dend'][m])       # Hay increases all conductances by 2 because resistance is also increased by 2

            cell._nrn_sections[sec_name].insert('CaDynamics_E2')
            for seg in cell._nrn_sections[sec_name]:
                seg.CaDynamics_E2.decay = 122
                seg.CaDynamics_E2.gamma = .0005096*1.1
                #seg.CaDynamics_E2.minCai = 5e-5
                # seg.CaDynamics_E2.decay = 460
                # seg.CaDynamics_E2.gamma = .000501


        return cell
    