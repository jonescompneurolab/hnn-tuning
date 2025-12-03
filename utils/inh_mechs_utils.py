import neuron
from neuron import h
from functools import partial
import numpy as np

class Mechs:

    def __init__(self, path_mod):
        neuron.load_mechanisms(path_mod)
        h.load_file("stdrun.hoc")


    def add_mechs(self,cell, params):

        mechs = params['soma'].keys()
        for i,m in enumerate(mechs):
            cell._nrn_sections['soma'].insert(m)
            for seg in cell._nrn_sections['soma']:
                if m == 'pas':
                    seg.pas.g = params['soma']['pas']['g']
                    seg.pas.e = params['soma']['pas']['e']
                else:
                    setattr(seg, 'gbar_'+m, params['soma'][m])

        return cell