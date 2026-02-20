"""Alpha-beta complex simulations"""

## set up environment

# HNN
from hnn_core import MPIBackend
from hnn_core import duecker_ET_model, simulate_dipole
from hnn_core.externals.mne import tfr_array_morlet

# other
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import scipy.signal as signal
from scipy.signal import find_peaks, butter, lfilter


# custom scripts and paths
import sys

sys.path.append('/users/kduecker/hnn-tuning/new_model_HNN/new_model_HNN/')
main_pth = '/users/kduecker/hnn-tuning/new_model_HNN/alpha_beta/'
out_pth = op.join(main_pth, 'sim_results')
os.makedirs(out_pth, exist_ok=True)

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
    params['default_ff_fb_seeds'] = (180, 120)  # seeds, this will be adjusted when running multiple trials
    params['weights_ampa_p'] = {'L2_basket': 0.0, 'L2_pyramidal': 2.5e-4, 'L5_pyramidal': 2.5e-4, 'L5_basket': 0.0} # proximal weights
    params['weights_ampa_d'] = {'L2_basket': 0, 'L5_pyramidal': 2.5e-4*2, 'L2_pyramidal': 2.5e-4*2} # distal weights
    params['syn_delays_p'] = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 1., 'L5_basket': 1.}
    params['syn_delays_d'] = {'L2_basket': .1, 'L5_pyramidal': .1, 'L2_pyramidal': .1}
    params['scale_factor'] = 10_000

    # PSD parameters
    params['nperseg_div'] = 4
    params['noverlap_div'] = 6

    params['fb_delay'] = 3

    params['n_procs'] = 10

    params['n_drive_cells_def'] = 10

    return params


def run_alpha_beta_complex_simulations(params, n_trials=1):
    """Run alpha-beta complex simulations"""

    net = duecker_ET_model()

    if params.get('prox_seed') is None:
        params['prox_seed'] = params['default_ff_fb_seeds'][0]
    if params.get('dist_seed') is None:
        params['dist_seed'] = params['default_ff_fb_seeds'][1]
    if params.get('n_drive_cells_prox') is None:
        params['n_drive_cells_prox'] = params['n_drive_cells_def']
    if params.get('n_drive_cells_dist') is None:
        params['n_drive_cells_dist'] = params['n_drive_cells_def']

    # feedforward drive
    net.add_bursty_drive('prox', tstart=50., burst_rate=10, burst_std=params['burst_std_prox'], numspikes=2,
                        spike_isi=10, n_drive_cells=params['n_drive_cells_prox'], location='proximal',
                        weights_ampa=params['weights_ampa_p'], synaptic_delays=params['syn_delays_p'], event_seed=params['prox_seed'])
    
    # feedback drive
    net.add_bursty_drive('dist', tstart=50.+params['fb_delay'], burst_rate=10, burst_std=params['burst_std_dist'], numspikes=2,
                        spike_isi=10, n_drive_cells=params['n_drive_cells_dist'], location='distal',
                        weights_ampa=params['weights_ampa_d'], synaptic_delays=params['syn_delays_d'], event_seed=params['dist_seed'])
    
    with MPIBackend(n_procs=params['n_procs'], mpi_cmd='mpiexec'):
        dpl = simulate_dipole(net, tstop=params['tstop'], bsl_cor='duecker', n_trials=n_trials)
    
    dpl = [d.scale(params['scale_factor']) for d in dpl]

    return dpl, net

def compute_psd(params, scale_dpl):
    """Compute power spectral density using Welch's method"""

    # compute power spectral density using Welch's method
    ts_vec = scale_dpl[0].times
    N = len(ts_vec)

    if params.get('nfft') is None:
        params['nfft'] = N

    P = np.zeros((len(scale_dpl), params['nfft']//2+1))
    f = np.zeros((len(scale_dpl), params['nfft']//2+1))
    for i, dpl in enumerate(scale_dpl):
        f[i], P[i] = signal.welch(dpl.data['agg'], fs=dpl.sfreq, window='hamming', nperseg=N/params['nperseg_div'], noverlap=N/params['noverlap_div'], nfft=params['nfft'], return_onesided=True, scaling='density')

        # normalize the power spectrum
        P[i] *= (f[i]/f[i][-1])

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
        timevec = np.arange(0.0, 7.0*sigma_t[fi], 1/scale_dpl[0].sfreq)
        # row wise merging (np.r_) -> real and imaginary part of the wavelet are symmetric (without this, we only get half a wavelet)
        # taken from MNE implementation of Morlet wavelets
        # see https://github.com/mne-tools/mne-python/blob/main/mne/time_frequency/tfr.py
        timevec = np.r_[-timevec[::-1], timevec[1:]]

    
        oscillation = np.exp(2*1j*np.pi*f*timevec)
        gaussian_envelope = np.exp(-(timevec**2)/(2*sigma_t[fi]**2))
        wavelets.append(A[fi]*oscillation*gaussian_envelope)

    #  convolve wavelets with signal
    tfr = np.zeros((len(scale_dpl), len(frequencies), len(scale_dpl[0].data['agg'])))
    for t, dpl in enumerate(scale_dpl):
        for fi, f in enumerate(frequencies):
            convolved = signal.convolve(dpl.data['agg'], wavelets[fi], mode='full')
            start = wavelets[fi].size // 2
            end = len(convolved) - (wavelets[fi].size // 2)
            # take the middle values
            tfr[t, fi, :] = np.abs(convolved[start:end])**2
    
    tfr = tfr.astype(np.float32)

    tfr = tfr.astype(np.float32)

    return frequencies, tfr

def compute_tfr_mne(scale_dpl):
    """Compute time-frequency representation using Morlet wavelets"""

    # padding
    total_pad_l = (3-scale_dpl[0].times[-1]/1000)*scale_dpl[0].sfreq
    pad_dpl = np.zeros((len(scale_dpl), int(np.round(total_pad_l+len(scale_dpl[0].times)))))
    times = np.linspace(-(3-scale_dpl[0].times[-1]/1000)/2, scale_dpl[0].times[-1]/1000+(3-scale_dpl[0].times[-1]/1000)/2, int(np.round(total_pad_l+len(scale_dpl[0].times))))

    for i, dpl in enumerate(scale_dpl):
        pad_dpl[i] = np.pad(dpl.data['agg'], (int(np.round(total_pad_l/2)), int(np.round(total_pad_l/2))), 'constant', constant_values=(0, 0))
    
    # compute time-frequency representation using Morlet wavelets
    sfreq = scale_dpl[0].sfreq
    frequencies = np.arange(4, 40, 1)  # frequencies from 4 to 40 Hz

    power = tfr_array_morlet(pad_dpl[:, np.newaxis, :], sfreq=sfreq,
                             freqs=frequencies, n_cycles=7, output='power')
    
    power = power[:, 0, :, :]  # remove the first two dimensions

    return frequencies, times, power, pad_dpl

def alpha_beta_ratio(dpl, dpl_type='hnn'):
    """Compute alpha-beta ratio from tfr"""

    frequencies, tfr = compute_tfr(dpl)

    if dpl_type=='hnn':
        tfr = tfr[:, :, :-1]
    sfreq = dpl[0].sfreq
    alpha_band = (frequencies >= 7) & (frequencies <= 14)
    beta_band = (frequencies >= 15) & (frequencies <= 29)
    length_sliding_window = int(0.1 * sfreq)


    # compute alpha-beta ratio
    power_bins = tfr.reshape(len(dpl), len(frequencies), length_sliding_window, -1) 
    alpha_power = np.mean(power_bins[:, alpha_band, :], axis=2).mean(axis=1)
    beta_power = np.mean(power_bins[:, beta_band, :], axis=2).mean(axis=1)

    ratio = alpha_power / beta_power

    # probablity of overlap in top third of alpha and beta power
    sort_alpha = np.argsort(alpha_power.reshape(-1))
    sort_beta = np.argsort(beta_power.reshape(-1))
    top3rd_ntrl = int(len(alpha_power.reshape(-1))/3)

    overlap = np.intersect1d(sort_alpha[-top3rd_ntrl:], sort_beta[-top3rd_ntrl:])
    prob_overlap = len(overlap)/top3rd_ntrl

    return ratio, alpha_power, beta_power, prob_overlap

def compute_symmetry_index(dpl, lowcut=7, highcut=29, prominence=0.01):

    """Compute symmetry index of the dipole waveform"""

    filt_dpl = _butter_bandpass_filter(dpl.data['agg'], lowcut, highcut, dpl.sfreq, order=2)
    peaks, _ = find_peaks(filt_dpl, prominence=prominence)
    troughs, _ = find_peaks(-filt_dpl, prominence=prominence)

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



def run_main(n_trials, n_procs, run_idx):

        
    filename = op.join(out_pth, f'alpha_beta_complex_simulation_delay_3ms{run_idx}.pkl')

    params = sim_analysis_params_default()
    params['n_procs'] = n_procs
    params['prox_seed'] = run_idx+100
    params['dist_seed'] = run_idx+60

    # set up dataframe
    df_dict = dict()

    dpl, _ = run_alpha_beta_complex_simulations(params, n_trials=n_trials)

    # alpha-beta ratio
    ab_ratio, alpha_power, beta_power, prob_overlap = alpha_beta_ratio(dpl)
    # symmetry index
    s_idx = np.zeros((n_trials,), dtype=object)
    for i, dip in enumerate(dpl):
        s, _ = compute_symmetry_index(dip, lowcut=7, highcut=29)
        s_idx[i] = s

    df = pd.DataFrame({
        'trial_index': range(n_trials),
        'alpha_beta_ratio': [ab_ratio[i, :] for i in range(n_trials)],
        'alpha_power': [alpha_power[i, :] for i in range(n_trials)],
        'beta_power': [beta_power[i, :] for i in range(n_trials)],
        'prob_overlap': [prob_overlap for _ in range(n_trials)],
        'symmetry_index': [s_idx[i] for i in range(n_trials)]
    })

    df.to_pickle(filename)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Alpha-beta complex, reproduce Jones et al., 2009.')

    parser.add_argument('--n_trials', type=int, default=1,
                        help='Number of trials to simulate (default: 1)')
    parser.add_argument('--n_procs', type=int, default=1,
                        help='Number of cores for MPI simulation (default: 1, run serially)')
    parser.add_argument('--run_idx', type=int, default=1,
                        help='Number of cores for MPI simulation (default: 1, run serially)')
    
    args = parser.parse_args()
    run_main(args.n_trials, args.n_procs, args.run_idx)

    


