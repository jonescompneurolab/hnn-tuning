"""mechs integrator utils"""
import neuron
from neuron import h
from functools import partial
import numpy as np

def Ih_condu_apical(gbar, offset, slope, factor_exp, x):

    return gbar * (offset +  slope * np.exp(x*factor_exp))

def distribute_gbar(gbar, slope, offset, x):

    return (slope * x + offset) * gbar

def get_iascone_params():

    iascone_model_gbar = {'soma': {'NaTs2_t': 0.926705, 
                                'SKv3_1': 0.102517,
                                'Nap_Et2': 0,
                                'Ca_HVA': 0.000374,
                                'Ca_LVAst': 0.000778,
                                'SK_E2': 0.099433,
                                'pas': 1.0 / 12000.0,
                                'Ih': 0.000080,
                                'Im': 0,},
                        'dend': {'NaTs2_t': 0.008009,
                                'SKv3_1': 0.000513,
                                'Ca_HVA': 0,
                                'Ca_LVAst': 0,
                                'pas': 1.0 / 12000.0,
                                'Ih': {'gbar': 0.000080, 'offset': -0.869600, 'slope':2.087+1, 'factor_exp':1/110},
                                'Im': 0.000740},
                        'e_pas': -75}
    
    return iascone_model_gbar


class Mechs:

    def __init__(self, path_mod):
        neuron.load_mechanisms(path_mod)
        h.load_file("stdrun.hoc")


    def add_mechs(self,cell, params):

        h.distance(sec = cell._nrn_sections['soma'])
        #params = get_rich_params()
        # apical_oblique as part of basal because it's the target of the proximal drive?
        basal_sections = ['basal_1', 'basal_2', 'basal_3', 'apical_oblique']

        for sec_name in basal_sections:
            cell._nrn_sections[sec_name].insert('pas')
            cell._nrn_sections[sec_name].insert('Ih')

            # add Sodium and potassium
            cell._nrn_sections[sec_name].insert('NaTa_t_l23')
            cell._nrn_sections[sec_name].insert('SKv3_1')

            for seg in cell._nrn_sections[sec_name]:
                seg.pas.g = params['soma']['pas']
                seg.pas.e = params['e_pas']
                seg.gbar_Ih = 0.000080

                setattr(seg, 'gbar_SKv3_1', params['soma']['SKv3_1']*0.1)
                setattr(seg, 'gbar_NaTa_t_l23', params['dend']['NaTa_t_l23']*2)
        
        # soma
        mechs = params['soma'].keys()
        for i,m in enumerate(mechs):
            cell._nrn_sections['soma'].insert(m)
            for seg in cell._nrn_sections['soma']:
                if m == 'pas':
                    seg.pas.g = params['soma']['pas']
                    seg.pas.e = params['e_pas']
                else:
                    setattr(seg, 'gbar_'+m, params['soma'][m])

        cell._nrn_sections['soma'].insert('CaDynamics_E2')
        for seg in cell._nrn_sections['soma']:
            seg.CaDynamics_E2.decay = 342.544232
            seg.CaDynamics_E2.gamma = 0.000533


        # apical sections
        apical_sections = ['apical_trunk', 'apical_1', 'apical_tuft']

        mechs = params['dend'].keys()
        h.distance(sec = cell._nrn_sections['soma'])
        for sec_name in apical_sections:
            for i,m in enumerate(mechs):
                for seg in cell._nrn_sections[sec_name]:
                    cell._nrn_sections[sec_name].insert(m)
                    distance = h.distance(seg.x, sec=seg.sec)
                    if m == 'pas':
                        seg.pas.g = params['dend']['pas']               # Hay increases all conductances by 2 because resistance is also increased by 2
                        seg.pas.e = params['e_pas']
                    
                    elif m == 'Ih':
                        gbar_fun = partial(Ih_condu_apical, params['dend'][m]['gbar'], params['dend'][m]['offset'], params['dend'][m]['slope'], params['dend'][m]['factor_exp'])
                        gbar = gbar_fun(distance)
                        setattr(seg, 'gbar_'+m, gbar)
                    else:
                        gbar_fun = partial(distribute_gbar, params['dend'][m], 0, 1)
                        gbar = gbar_fun(distance)
                        setattr(seg,'gbar_'+m, gbar)

        return cell
    