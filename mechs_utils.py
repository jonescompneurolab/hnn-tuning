"""mechs integrator utils"""
import neuron
from neuron import h
from functools import partial
import numpy as np

def Ih_condu_apical(gsoma, x):

    return gsoma * (-0.8696 + 2.087 * np.exp(x/323))

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


    def add_mechs(self,cell, mechs, hotzone=[650, 950]):

        params = get_hay_params()
        # basal sections (passive only)
        basal_sections = ['basal_1', 'basal_2', 'basal_3']

        for sec_name in basal_sections:
            cell._nrn_sections[sec_name].insert('pas')
            cell._nrn_sections[sec_name].insert('Ih')
            for seg in cell._nrn_sections[sec_name]:
                seg.pas.g = .0000467
                seg.pas.e = params['e_pas']
                seg.gbar_Ih = .002
        
        # soma
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
                    elif m == 'CaDynamics_E2':
                        seg.CaDynamics_E2.decay = 460
                        seg.CaDynamics_E2.gamma = .000501
                    else:
                        setattr(seg, 'gbar_'+m, params['soma'][m])

        # apical sections
        apical_sections = ['apical_trunk', 'apical_oblique', 'apical_1', 'apical_2', 'apical_tuft']
        h.distance(sec = cell._nrn_sections['soma'])
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
                        elif m == 'CaDynamics_E2':
                            seg.CaDynamics_E2.decay = 122
                            seg.CaDynamics_E2.gamma = .000509
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
                                setattr(seg, 'gbar_'+m, params['dend'][m]*2)       # Hay increases all conductances by 2 because resistance is also increased by 2


        return cell
    