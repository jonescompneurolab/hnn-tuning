"""mechs integrator utils"""
import neuron
from neuron import h
from functools import partial
import numpy as np

def Ih_condu_apical(gsoma, x):

    sl = 1#3.6161
    return gsoma * (-0.8696 + 2.087 * np.exp(sl*(x/323)))

def get_rich_params():

    rich_model_gbar = {'soma': {'NaTs2_t': 2.1, 
                                'SKv3_1': .04,
                                'Nap_Et2': 1.9e-6,
                                'Ca_HVA': 5.7e-9,
                                'Ca_LVAst': .001,
                                'SK_E2': 2.45e-9,
                                'pas': .338e-4,
                                'Ih': 5.14e-5,
                                'K_Pst': .065,
                                'K_Tst': 2.e-5},
                        'dend': {'NaTa_t': 0.001,
                                'SKv3_1': .04,
                                'Ca_HVA': .00056,#5.7e-9,
                                'Ca_LVAst': .001,
                                'SK_E2': 2.45e-9,
                                'pas': 0.0000589,
                                'Ih': partial(Ih_condu_apical, 5.14e-5),
                                'Im': 2e-4,
                                'K_Tst': 2.e-5},
                        'e_pas': -84.4}
    
    # rich_model_gbar = {'soma': {'NaTs2_t': 20_400e-4, 
    #                             'SKv3_1': 6_930e-4,
    #                             'Nap_Et2': 17.2e-4,
    #                             'Ca_HVA': 9.92e-04,
    #                             'Ca_LVAst': 34.3e-4,
    #                             'SK_E2': 441e-4,
    #                             'pas': .338e-4,
    #                             'Ih': 1e-4,
    #                             'Im': 0,
    #                             'K_Pst': 22.3e-4,
    #                             'K_Tst': 812e-4},
    #                     'dend': {'NaTa_t': 0.0213,
    #                             'SKv3_1': 0.000261,
    #                             'Nap_Et2': 0,
    #                             'Ca_HVA': 2.78e-5,
    #                             'Ca_LVAst': 93.5e-6,
    #                             'SK_E2': 0.0012,
    #                             'pas': 0.0000589,
    #                             'Ih': partial(Ih_condu_apical, 1e-4),
    #                             'Im': 0.0000675,
    #                             'K_Pst': 0,
    #                             'K_Tst': 0},
    #                     'e_pas': -84.4}
    
    return rich_model_gbar


class Mechs:

    def __init__(self, path_mod):
        neuron.load_mechanisms(path_mod)
        h.load_file("stdrun.hoc")


    def add_mechs(self,cell, hotzone=[650, 950]):

        params = get_rich_params()
        # basal sections (passive only)
        basal_sections = ['basal_1', 'basal_2', 'basal_3']

        # DefineProcs.hoc actually suggests that several mechanisms are added to basal dendrite
        for sec_name in basal_sections:
            cell._nrn_sections[sec_name].insert('pas')
            cell._nrn_sections[sec_name].insert('Ih')
            for seg in cell._nrn_sections[sec_name]:
                seg.pas.g = 1.75e-5
                seg.pas.e = params['e_pas']
                seg.gbar_Ih = 5.14e-5
        
        # soma
        mechs = params['soma'].keys()
        for i,m in enumerate(mechs):
            if m == 'NaT':
                cell._nrn_sections['soma'].insert('NaTs2_t')
                for seg in cell._nrn_sections['soma']:
                    setattr(seg, 'gbar_NaTs2_t', params['soma']['NaTs2_t']) 
            else: 
                cell._nrn_sections['soma'].insert(m)
                for seg in cell._nrn_sections['soma']:
                    if m == 'pas':
                        seg.pas.g = params['soma']['pas']
                        seg.pas.e = params['e_pas']
                    else:
                        setattr(seg, 'gbar_'+m, params['soma'][m])

        cell._nrn_sections['soma'].insert('CaDynamics_E2')
        for seg in cell._nrn_sections['soma']:
            seg.CaDynamics_E2.decay = 460
            seg.CaDynamics_E2.gamma = .000501


        # apical sections
        apical_sections = ['apical_trunk', 'apical_oblique', 'apical_1', 'apical_2', 'apical_tuft']
        h.distance(sec = cell._nrn_sections['soma'])

        mechs = params['dend'].keys()
        print(mechs)
        for sec_name in apical_sections:
            for i,m in enumerate(mechs):
                if m == 'NaT':
                    cell._nrn_sections[sec_name].insert('NaTa_t')
                    for seg in cell._nrn_sections[sec_name]:
                        setattr(seg, 'gbar_NaTa_t', params['dend']['NaTa_t']) 
                else: 
                    cell._nrn_sections[sec_name].insert(m)
                    for seg in cell._nrn_sections[sec_name]:
                        if m == 'pas':
                            seg.pas.g = params['dend']['pas']               # Hay increases all conductances by 2 because resistance is also increased by 2
                            seg.pas.e = params['e_pas']
                        
                        elif m == 'Ih':
                            distance = h.distance(seg.x, sec=seg.sec)
                            gbar = params['dend']['Ih'](distance)
                            setattr(seg, 'gbar_'+m, gbar)
                        
                        else:
                            distance = h.distance(seg.x, sec=seg.sec)
                            if (distance >= hotzone[0]) and (distance < hotzone[1]):
                                if m == 'Ca_HVA':
                                    setattr(seg, 'gbar_'+m, params['dend'][m]*10)
                                elif m == 'Ca_LVAst':
                                    setattr(seg, 'gbar_'+m, params['dend'][m]*100)
                            else:
                                setattr(seg, 'gbar_'+m, params['dend'][m])       # Hay increases all conductances by 2 because resistance is also increased by 2

            cell._nrn_sections[sec_name].insert('CaDynamics_E2')
            for seg in cell._nrn_sections[sec_name]:
                seg.CaDynamics_E2.decay = 122
                seg.CaDynamics_E2.gamma = .000509
                # seg.CaDynamics_E2.decay = 460
                # seg.CaDynamics_E2.gamma = .000501


        return cell
    