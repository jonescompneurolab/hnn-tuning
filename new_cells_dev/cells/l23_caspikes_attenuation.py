"""" L2/3 pyramidal cell plots """

import neuron
from neuron import h
import matplotlib.pyplot as plt
from neuron.units import ms, mV
import numpy as np
pth_mod = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/hnn-core/hnn_core/mod'
figpth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/new_cells_dev/figures/l23_tuning/'
import os
os.makedirs(figpth, exist_ok=True)
plt.rcParams['svg.fonttype'] = 'none'

import sys
sys.path.append('/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/utils')
from cell_builder_utils import l2_params, _get_dends, _get_pyr_soma, Cell
from functools import partial
from l5_mechs_utils import linear_gbar, Ih_condu_hay
from hnn_core.viz import plot_cell_morphology
from l23_plot_utils import plot_conductances, get_colormap, plot_ca_spike, plot_ca_spike4x2
from misc_utils import plot_attenuation

from matplotlib import colormaps
from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# Plot conductances
# fig = plot_conductances()
# fig.savefig(os.path.join(figpth, 'l23_conductances_inp_res_membrane_time.pdf'))
# fig.savefig(os.path.join(figpth, 'l23_conductances_inp_res_membrane_time.svg'))

# fig.savefig(os.path.join(figpth, 'l23_conductances_inp_res_membrane_time.png'))
# plt.close('all')

# Plot attenuation from soma
_, cols_dist, _, _, _ = get_colormap()
# plt.close('all')
fig = plot_attenuation(cols_dist)
fig.savefig(os.path.join(figpth, 'summary_attenuation_soma_dend.png'), dpi=300)
fig.savefig(os.path.join(figpth, 'summary_attenuation_soma_dend.pdf'), dpi=300)
plt.close()

# Plot attenuation from tuft
fig = plot_attenuation(cols_dist, ref_sect='apical_tuft', model_name='l2pyr', epsp=True, amp_hyperpol=-2, amp_spike=.5)
fig.savefig(os.path.join(figpth, 'summary_attenuation_dend_soma.png'), dpi=300)
fig.savefig(os.path.join(figpth, 'summary_attenuation_dend_soma.pdf'), dpi=300)
plt.close()

# # Plot Ca-spike

# fig = plot_ca_spike()

# fig.savefig(os.path.join(figpth, 'ca_spikes.png'), dpi=300)
# fig.savefig(os.path.join(figpth, 'ca_spikes.pdf'), dpi=300)
# fig.savefig(os.path.join(figpth, 'ca_spikes.svg'), dpi=300)

# fig = plot_ca_spike4x2()

# fig.savefig(os.path.join(figpth, 'ca_spikes_4x2.png'), dpi=300)
# fig.savefig(os.path.join(figpth, 'ca_spikes_4x2.pdf'), dpi=300)
# fig.savefig(os.path.join(figpth, 'ca_spikes_4x2.svg'), dpi=300)


