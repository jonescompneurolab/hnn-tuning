"""Alpha-beta complex simulations"""

## set up environment
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import random
import scipy.signal as signal
from scipy.signal import find_peaks, butter, lfilter

# HNN
import hnn_core
from hnn_core import MPIBackend, JoblibBackend
from hnn_core.dipole import _rmse
from hnn_core.params import _short_name

from hnn_core import duecker_2025_ET_model, simulate_dipole, jones_2009_model
from hnn_core.externals.mne import tfr_array_morlet
from hnn_core.viz import plot_dipole, plot_psd, plot_tfr_morlet

# custom scripts and paths
import sys
sys.path.append('/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/new_model_HNN/')
main_pth = '/Users/katharinaduecker/Documents/projects_brown/hnn-tuning/new_model_HNN/alpha_beta/'
out_pth = op.join(main_pth, 'sim_results')
os.makedirs(out_pth, exist_ok=True)

from optim_utils import set_params


## Set up parameters for simulations

def _butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def sim_analysis_params_default():
    """Set up simulation and analysis parameters"""

    params = dict()

    # simulation parameters
    params['tstop'] = 1000.  # ms
    params['burst_std_prox'] = 20 # Jones 2009: input bursts variance = 400
    params['burst_std_dist'] = 20
    params['default_ff_fb_seeds'] = (180, 120)  # seeds
    params['weights_ampa_p'] = {'L2_basket': 0.0, 'L2_pyramidal': 2.5e-4, 'L5_pyramidal': 2.5e-4, 'L5_basket': 0.0} # proximal weights
    params['weights_ampa_d'] = {'L2_basket': 0, 'L5_pyramidal': 2.5e-4*4, 'L2_pyramidal': 2.5e-4*4} # distal weights
    params['syn_delays_p'] = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 1., 'L5_basket': 1.}
    params['syn_delays_d'] = {'L2_basket': .1, 'L5_pyramidal': 1., 'L2_pyramidal': .1}
    params['scale_factor'] = 5_000

    # PSD parameters
    params['nperseg_div'] = 4
    params['noverlap_div'] = 6

    params['fb_delay'] = 10

    params['n_procs'] = 10

    params['n_drive_cells_def'] = 10

    return params


def run_alpha_beta_complex_simulations(params):
    """Run alpha-beta complex simulations"""

    net = duecker_2025_ET_model()

    if params.get('prox_seed') is None:
        params['prox_seed'] = params['default_ff_fb_seeds'][0]
    if params.get('dist_seed') is None:
        params['dist_seed'] = params['default_ff_fb_seeds'][1]
    if params.get('n_drive_cells_prox') is None:
        params['n_drive_cells_prox'] = params['n_drive_cells_def']
    if params.get('n_drive_cells_dist') is None:
        params['n_drive_cells_dist'] = params['n_drive_cells_def']

    # feedforward drive
    net.add_bursty_drive('alpha_prox', tstart=50., burst_rate=10, burst_std=params['burst_std_prox'], numspikes=2,
                        spike_isi=10, n_drive_cells=params['n_drive_cells_prox'], location='proximal',
                        weights_ampa=params['weights_ampa_p'], synaptic_delays=params['syn_delays_p'], event_seed=params['prox_seed'])
    
    # feedback drive
    net.add_bursty_drive('beta_dist', tstart=50.+params['fb_delay'], burst_rate=10, burst_std=params['burst_std_dist'], numspikes=2,
                        spike_isi=10, n_drive_cells=params['n_drive_cells_dist'], location='distal',
                        weights_ampa=params['weights_ampa_d'], synaptic_delays=params['syn_delays_d'], event_seed=params['dist_seed'])
    
    with MPIBackend(n_procs=params['n_procs'], mpi_cmd='mpiexec'):
        dpl = simulate_dipole(net, tstop=params['tstop'], bsl_cor='calcium')[0]
    
    dpl.scale(params['scale_factor'])

    return dpl, net

def compute_psd(params, scale_dpl):
    """Compute power spectral density using Welch's method"""

    # compute power spectral density using Welch's method
    ts_vec = scale_dpl.times
    N = len(ts_vec)

    if params.get('nfft') is None:
        params['nfft'] = N

    f, P = signal.welch(scale_dpl.data['agg'], fs=scale_dpl.sfreq, window='hamming', nperseg=N/params['nperseg_div'], noverlap=N/params['noverlap_div'], nfft=params['nfft'], return_onesided=True, scaling='density')

    # normalize the power spectrum
    P *= (f/f[-1])

    return f, P

def compute_tfr(scale_dpl):
    """Compute time-frequency representation using Morlet wavelets"""

    frequencies = np.arange(1, 40, 1)  # frequencies
    m = 7                               # number of cycles

    sigma_t = m/(2*np.pi*frequencies)   # standard deviation of the Gaussian envelope
    A = 1/(sigma_t*np.sqrt(2*np.pi))    # normalization factor according to Jones 2009

    # create wavelets
    wavelets = []
    for fi, f in enumerate(frequencies):
        # timevec is different for each frequency
        timevec = np.arange(0.0, 7.0*sigma_t[fi], 1/scale_dpl.sfreq)
        # row wise merging (np.r_) -> real and imaginary part of the wavelet are symmetric (without this, we only get half a wavelet)
        # taken from MNE implementation of Morlet wavelets
        # see https://github.com/mne-tools/mne-python/blob/main/mne/time_frequency/tfr.py
        timevec = np.r_[-timevec[::-1], timevec[1:]]

    
        oscillation = np.exp(2*1j*np.pi*f*timevec)
        gaussian_envelope = np.exp(-(timevec**2)/(2*sigma_t[fi]**2))
        wavelets.append(A[fi]*oscillation*gaussian_envelope)

    #  convolve wavelets with signal
    tfr = np.zeros((len(frequencies), len(scale_dpl.data['agg'])))
    for fi, f in enumerate(frequencies):
        convolved = signal.convolve(scale_dpl.data['agg'], wavelets[fi], mode='full')
        start = wavelets[fi].size // 2
        end = len(convolved) - (wavelets[fi].size // 2)
        # take the middle values
        tfr[fi, :] = np.abs(convolved[start:end])**2

    return frequencies, tfr

def compute_tfr_mne(scale_dpl):
    """Compute time-frequency representation using Morlet wavelets"""

    # padding
    # padding for TFR
    pad_dpl = scale_dpl.copy()
    total_pad_l = (3-scale_dpl.times[-1]/1000)*scale_dpl.sfreq

    pad_dpl.data['agg'] = np.pad(pad_dpl.data['agg'], (int(np.round(total_pad_l/2)), int(np.round(total_pad_l/2))), 'constant', constant_values=(0, 0))
    pad_dpl.times = np.pad(pad_dpl.times, (int(np.round(total_pad_l/2)), int(np.round(total_pad_l/2))), 'constant', constant_values=(pad_dpl.times[0]-int(np.round(total_pad_l/2))/pad_dpl.sfreq, pad_dpl.times[-1]+int(np.round(total_pad_l/2))/scale_dpl.sfreq))
    pad_dpl.data['L5'] = np.pad(pad_dpl.data['L5'], (int(np.round(total_pad_l/2)), int(np.round(total_pad_l/2))), 'constant', constant_values=(0, 0))
    pad_dpl.data['L2'] = np.pad(pad_dpl.data['L2'], (int(np.round(total_pad_l/2)), int(np.round(total_pad_l/2))), 'constant', constant_values=(0, 0))


    # compute time-frequency representation using Morlet wavelets
    sfreq = scale_dpl.sfreq
    frequencies = np.arange(4, 40, 1)  # frequencies from 4 to 40 Hz

    power = tfr_array_morlet(pad_dpl.data['agg'][np.newaxis, np.newaxis, :], sfreq=sfreq,
                             freqs=frequencies, n_cycles=7, output='power')
    
    power = power[0, 0]  # remove the first two dimensions
    times = pad_dpl.times

    return frequencies, times, power, pad_dpl

def alpha_beta_ratio(dpl):
    """Compute alpha-beta ratio from tfr"""

    frequencies, tfr = compute_tfr(dpl)
    tfr = tfr[:, :-1]
    sfreq = dpl.sfreq
    alpha_band = (frequencies >= 7) & (frequencies <= 14)
    beta_band = (frequencies >= 15) & (frequencies <= 29)
    length_sliding_window = int(0.1 * sfreq)

    # compute alpha-beta ratio
    power_bins = tfr.reshape(len(frequencies), length_sliding_window, -1, order='F') 
    alpha_power = np.mean(power_bins[alpha_band, :], axis=1).mean(axis=0)
    beta_power = np.mean(power_bins[beta_band, :], axis=1).mean(axis=0)

    ratio = alpha_power / beta_power

    return ratio, alpha_power, beta_power

def compute_symmetry_index(dpl, lowcut=7, highcut=29):

    """Compute symmetry index of the dipole waveform"""

    filt_dpl = _butter_bandpass_filter(dpl.data['agg'], lowcut, highcut, dpl.sfreq, order=2)
    peaks, _ = find_peaks(filt_dpl, prominence=0.01)
    troughs, _ = find_peaks(-filt_dpl, prominence=0.01)

    if len(peaks) == len(troughs):
        if troughs[0] < peaks[0]:
            sind = (np.abs(filt_dpl[peaks[:-1]]) - np.abs(filt_dpl[troughs[1:]]))/(np.abs(filt_dpl[peaks[:-1]]) + np.abs(filt_dpl[troughs[1:]]))
        else:
            sind = (np.abs(filt_dpl[peaks]) - np.abs(filt_dpl[troughs]))/(np.abs(filt_dpl[peaks]) + np.abs(filt_dpl[troughs]))
    elif len(peaks) > len(troughs):
         sind = (np.abs(filt_dpl[peaks[int(len(peaks)-len(troughs)):]]) - np.abs(filt_dpl[troughs]))/(np.abs(filt_dpl[peaks[int(len(peaks)-len(troughs)):]]) + np.abs(filt_dpl[troughs]))
    elif len(peaks) < len(troughs):

        peak_diff = int(len(troughs)-len(peaks))
        if troughs[0] < peaks[0]:
            sind = (np.abs(filt_dpl[peaks]) - np.abs(filt_dpl[troughs[peak_diff:]]))/(np.abs(filt_dpl[peaks]) + np.abs(filt_dpl[troughs[peak_diff:]]))
        else:
            sind = (np.abs(filt_dpl[peaks]) - np.abs(filt_dpl[troughs[:-peak_diff]]))/(np.abs(filt_dpl[peaks]) + np.abs(filt_dpl[troughs[:-peak_diff]]))

    # return symmetry index and filtered dipole with peaks and troughs
    return sind, (filt_dpl, peaks, troughs)



def run_main(seed, job_id, n_trials, n_procs):

    random.seed(seed)
    filename = op.join(out_pth, f'alpha_beta_complex_simulation_job{job_id}_seed{seed}.pkl')

    params = sim_analysis_params_default()
    params['n_procs'] = n_procs

    # set up dataframe
    df_dict = dict()
    df_dict['trial'] = []

    df_dict['seed_prox'] = []
    df_dict['seed_dist'] = []
    df_dict['alpha_power'] = []
    df_dict['beta_power'] = []
    df_dict['alpha_beta_ratio'] = []
    df_dict['symmetry_index'] = []

    for n in range(n_trials):

        df_dict['trial'].append(n)
        
        # change seeds and make sure they are different
        params['prox_seed'] = random.randint(0,10000)
        params['dist_seed'] = random.randint(0,10000)
        while params['dist_seed'] == params['prox_seed']:
            params['dist_seed'] = random.randint(0,10000)

        df_dict['seed_prox'].append(params['prox_seed'])
        df_dict['seed_dist'].append(params['dist_seed'])

        dpl, _ = run_alpha_beta_complex_simulations(params)

        # alpha-beta ratio
        ab_ratio, alpha_power, beta_power = alpha_beta_ratio(dpl)
        # symmetry index
        s_idx, _ = compute_symmetry_index(dpl, lowcut=7, highcut=29)

        df_dict['alpha_beta_ratio'].append(ab_ratio)
        df_dict['alpha_power'].append(alpha_power)
        df_dict['beta_power'].append(beta_power)
        df_dict['symmetry_index'].append(s_idx)

    df = pd.DataFrame(df_dict)
    df.to_pickle(filename)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Alpha-beta complex, reproduce Jones et al., 2009.')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (ensures different seeds for drives default: 42)')
    parser.add_argument('--job_id', type=int, default=0,
                        help='Job ID to keep track of parallel processing (default: 0)')
    parser.add_argument('--n_trials', type=int, default=1,
                        help='Number of trials to simulate (default: 1)')
    parser.add_argument('--n_procs', type=int, default=1,
                        help='Number of cores for MPI simulation (default: 1, run serially)')
    parser.add_argument('--ratios', action='store_true',
                        help='Run alpha-beta ratio simulations for different parameters (default: False)')
    
    args = parser.parse_args()
    run_main(args.seed, args.job_id, args.n_trials, args.n_procs)

    


