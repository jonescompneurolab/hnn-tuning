import neuron
from neuron import h
import matplotlib.pyplot as plt
from neuron.units import ms, mV
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import pickle
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import chirp

from scipy.signal import windows

import os
import os.path as op

pth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning'
pth_mod = op.join(pth, 'hnn-core/hnn_core/mod')
figpth = op.join(pth, 'new_cells_dev/figures/l5_ET_tuning/')
os.makedirs(figpth, exist_ok=True)
import sys
sys.path.append(op.join(pth, 'utils'))

from l5_plot_utils import get_colormap, fig_caspike_critical_frequency, fig_caspike_coincidence, fig_electro_measures, fig_conductances
from misc_utils import plot_attenuation

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# Set global font sizes
plt.rcParams['font.size'] = plt.rcParams['axes.titlesize']= plt.rcParams['axes.labelsize'] = plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = plt.rcParams['legend.fontsize'] = 12          # Default text size

# get colormaps
im, cols_dist, col_basal, cols_experiments = get_colormap()

# Plot conductances
# fig = fig_conductances(col_basal)
# fig.savefig(figpth+ 'l5_conductances_basal_dend.pdf')
# fig.savefig(figpth+ 'l5_conductances_basal_dend.svg')

# # Plot Ca spike critical frequency
# fig = fig_caspike_critical_frequency(cols_experiments)
# fig.savefig(figpth+'soma_vs_apical_critfreq.pdf', format='pdf', dpi=300)
# fig.savefig(figpth+'soma_vs_apical_critfreq.svg', format='svg', dpi=300)

# # # Plot Ca spike coincidence detection
# fig = fig_caspike_coincidence(cols_experiments)
# fig.savefig(figpth+'caspike_coincidence_detection.pdf', format='pdf', dpi=300)
# fig.savefig(figpth+'caspike_coincidence_detection.svg', format='svg', dpi=300)

# # Plot different measures related to impedance in electric input
# fig,distance = fig_electro_measures()
# fig.savefig(figpth+'l5_electro_measures.pdf', format='pdf', dpi=300)
# fig.savefig(figpth+'l5_electro_measures.svg', format='svg', dpi=300)

# Plot attenuation from soma
_, cols_dist, _, _ = get_colormap()
plt.close('all')
fig = plot_attenuation(cols_dist, model_name='l5pyr')
fig.savefig(op.join(figpth, 'summary_attenuation_soma_dend.png'), dpi=300)
fig.savefig(op.join(figpth, 'summary_attenuation_soma_dend.pdf'), dpi=300)
plt.close()

# Plot attenuation from tuft
fig = plot_attenuation(cols_dist, ref_sect='apical_tuft', model_name='l5pyr', epsp=True, amp_hyperpol=-2, amp_spike=.5)
fig.savefig(op.join(figpth, 'summary_attenuation_dend_soma.png'), dpi=300)
fig.savefig(op.join(figpth, 'summary_attenuation_dend_soma.pdf'), dpi=300)
plt.close()

