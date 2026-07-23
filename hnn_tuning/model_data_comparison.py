"""Helpers for comparing model cells against experimental electrophysiology.

Shared by the single-cell comparison scripts (``*_vsAllen``,
``l5_ETvsKalmbach2021``). Pure numpy/pandas — no NEURON import — so it loads
quickly and can be imported from any environment.
"""

import numpy as np
import pandas as pd


def exp_func(t, tau, A, C):
    """Exponential rise, fit to a hyperpolarising step to extract tau.

    Used on both sides of the comparison: to fit the membrane time constant of
    the model cell's response and of the experimental sweeps.
    """
    return A * (1 - np.exp(-t / tau)) + C


def _parse_array(s):
    """Restore a numpy array from its CSV string form, e.g. '[-0.11 -0.09]'.

    Feature tables are stored as CSV, which turns array-valued cells (such as
    ``sag_stim_amp``) into strings; this converts them back. NaN (a failed
    specimen) passes through unchanged.
    """
    if not isinstance(s, str):
        return s
    return np.array([float(x) for x in s.strip('[]').split()])


def load_ephys_csv(path):
    """Load an experimental ephys feature table, restoring array-valued columns.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a feature CSV written by ``allen_get_ephys_data.py`` (specimen
        id as the index column).

    Returns
    -------
    pandas.DataFrame
        The table, with ``sag_stim_amp`` converted back to arrays if present.
    """
    df = pd.read_csv(path, index_col=0)
    if 'sag_stim_amp' in df.columns:
        df['sag_stim_amp'] = df['sag_stim_amp'].apply(_parse_array)
    return df
