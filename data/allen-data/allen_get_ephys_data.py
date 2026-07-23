"""Extract electrophysiological features from human in vitro soma recordings.

Requires the allensdk environment (see requirements-allen.txt), which must be
separate from the main hnn-tuning environment.

Usage
-----
    # one cell type (here l2 pyramidal)
    python allen_get_ephys_data.py --cell-type l2 --data-dir <path_to_data>

    # all three cell types
    python allen_get_ephys_data.py --all --data-dir <path_to_data>

Output
------
For each cell type, a .csv and a .pkl saved to data/allen-data/:

    interneuron/ephys_inh.csv   
    l23/ephys_l2.csv            
    l23/ephys_l3.csv           

Author: Katharina Duecker <katharina.duecker@gmail.com>
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features

# The csv files containing the specimen details
SPECIMEN_DETAILS = "cell_types_specimen_details.csv"

# The three selections that were code cells [5], [6] and [7] of the notebook.
# `query` is applied to the specimen details table; `subdir` and `stem` say
# where the result is written.
CELL_TYPES = {
    "interneuron": {
        "subdir": "interneuron",
        "stem": "ephys_inh",
        "dendrite_type": "aspiny",
        "layer": None,
    },
    "l2": {
        "subdir": "l23",
        "stem": "ephys_l2",
        "dendrite_type": "spiny",
        "layer": "2",
    },
    "l3": {
        "subdir": "l23",
        "stem": "ephys_l3",
        "dendrite_type": "spiny",
        "layer": "3",
    },
}


# --------------------------------------------------------------------------
# helpers -- local to this script, since nothing else uses them
# --------------------------------------------------------------------------
def _exp_func_mem_time(t, tau, A, C):
    """Exponential used to fit the membrane time constant."""
    return A * (1 - np.exp(-t / tau)) + C


def _select_specimen_ids(df, cell_type):
    """Return the specimen ids matching `cell_type` from the details table."""
    spec = CELL_TYPES[cell_type]
    mask = (df["donor__species"] == "Homo Sapiens") & (
        df["tag__dendrite_type"] == spec["dendrite_type"]
    )
    if spec["layer"] is not None:
        mask &= df["structure__layer"] == spec["layer"]
    return df["specimen__id"][mask]


def _features_for_specimen(ctc, specimen_id):
    """Compute the feature dict for a single specimen.

    Returns a dict of features. On any failure, returns the same NaN-filled
    dict the notebook produced.
    """
    features = {"index": specimen_id}

    # download the ephys data and sweep metadata
    data_set = ctc.get_ephys_data(specimen_id)
    sweeps = ctc.get_ephys_sweeps(specimen_id)

    # group the sweeps by stimulus
    sweep_numbers = defaultdict(list)
    for sweep in sweeps:
        sweep_numbers[sweep["stimulus_name"]].append(sweep["sweep_number"])

    cell_features = extract_cell_features(
        data_set,
        sweep_numbers["Ramp"],
        sweep_numbers["Short Square"],
        sweep_numbers["Long Square"],
    )

    long_square = cell_features["long_squares"]
    long_df = pd.DataFrame(long_square["sweeps"])

    # membrane potential
    features["RMP"] = np.mean(long_df.v_baseline.values)

    # sag
    features["sag"] = np.mean(long_df["sag"][long_df["stim_amp"] < 0].values)

    # amplitudes used to extract the membrane time constant tau
    features["sag_stim_amp"] = (
        long_df["stim_amp"][long_df["stim_amp"] < 0].values / 1000
    )

    # ISIs for stimuli with positive amplitude (depolarising)
    features["max_amp_isi"] = np.max(long_df["stim_amp"]) / 1000

    # action potential features
    short_square = cell_features["short_squares"]
    first_spike = short_square["common_amp_sweeps"][0]["spikes"][0]
    features["upstroke_downstroke"] = first_spike["upstroke_downstroke_ratio"]
    features["ap_width"] = first_spike["width"] * 1000
    features["ap_threshold"] = first_spike["threshold_v"]

    sweeps_meta = {
        sweep: data_set.get_sweep_metadata(sweep)
        for sweep in data_set.get_experiment_sweep_numbers()
    }
    sweeps_df = pd.DataFrame.from_dict(sweeps_meta, orient="index")

    # --- depolarising long square sweeps: firing rate and ISIs -----------
    sweeps_long = sweeps_df[
        (sweeps_df["aibs_stimulus_name"] == b"Long Square")
        & (sweeps_df["aibs_stimulus_amplitude_pa"] > 0)
    ]

    num_spks = np.zeros_like(sweeps_long.index)   # number of spikes
    ampl = np.zeros_like(sweeps_long.index)       # current injection amplitude
    first_isi = np.zeros_like(sweeps_long.index)  # ISI of first spikes
    mean_isi = np.zeros_like(sweeps_long.index)   # ISI of following spikes

    for i, s in enumerate(sweeps_long.index):
        sweep = data_set.get_sweep(s)

        v = sweep["response"]
        pk = find_peaks(v, height=0)
        num_spks[i] = len(pk[0])
        ampl[i] = sweeps_long["aibs_stimulus_amplitude_pa"][s]

        dt = 1.0e3 / sweep["sampling_rate"]
        t = np.arange(len(v)) * dt

        if len(pk[0]) > 1:
            first_isi[i] = np.diff(t[pk[0]])[0]
            mean_isi[i] = np.mean(np.diff(t[pk[0]]))

    features["FR_slope"] = np.polyfit(
        ampl[num_spks > 0], num_spks[num_spks > 0], 1
    )[0]

    # to avoid outliers, take the median of first and mean ISI 
    features["first_isi"] = np.median(first_isi[first_isi > 0])
    features["mean_isi"] = np.median(first_isi[mean_isi > 0])

    # --- hyperpolarising long square sweeps: membrane time constant ------
    sweeps_long = sweeps_df[
        (sweeps_df["aibs_stimulus_name"] == b"Long Square")
        & (sweeps_df["aibs_stimulus_amplitude_pa"] < 0)
    ]

    tau_ms = np.zeros_like(sweeps_long.index)
    for i, s in enumerate(sweeps_long.index):
        sweep = data_set.get_sweep(s)

        t_max = len(sweep["response"]) / sweep["sampling_rate"] * 1000  # s -> ms
        timevec = np.linspace(0, t_max, len(sweep["response"]))

        stim_onset = np.where(sweep["stimulus"] < 0)[0][0]
        stim_offset = (
            np.where(sweep["stimulus"][stim_onset:] >= 0)[0][0] + stim_onset
        )

        t_rise = timevec[stim_onset:stim_offset] - timevec[stim_onset]
        v_rise = sweep["response"][stim_onset:stim_offset]

        # baseline correction
        delta_v = np.mean(v_rise) - np.mean(sweep["response"][:stim_onset])
        popt, _ = curve_fit(
            _exp_func_mem_time,
            t_rise,
            v_rise,
            p0=[10, abs(delta_v), v_rise[0]],
        )
        tau_ms[i] = popt[0]

    features["tau"] = np.mean(tau_ms)

    return features


def _nan_features(specimen_id):
    """The fallback dict the notebook produced when extraction failed."""
    return {
        "index": specimen_id,
        "RMP": float("nan"),
        "sag": float("nan"),
        "mean_isi": float("nan"),
        "first_isi": float("nan"),
        "upstroke_downstroke": float("nan"),
        "ap_width": float("nan"),
        "ap_threshold": float("nan"),
        "FR_slope": float("nan"),
        "tau": float("nan")
    }


def extract_ephys(ctc, specimen_ids):
    """Extract features for every specimen, tolerating individual failures."""
    ephys_features = dict()
    n_total = len(specimen_ids)

    for i, specimen_id in enumerate(specimen_ids.values, start=1):
        print(f"  [{i}/{n_total}] specimen {specimen_id}", flush=True)
        try:
            ephys_features[specimen_id] = _features_for_specimen(
                ctc, specimen_id
            )
        except Exception as exc:  # noqa: BLE001 -- notebook used a bare except
            print(f"      failed ({type(exc).__name__}: {exc})", flush=True)
            ephys_features[specimen_id] = _nan_features(specimen_id)

    return ephys_features


# --------------------------------------------------------------------------
# extract ephys features for each cell type
# --------------------------------------------------------------------------
def extract_for_cell_type(ctc, df, cell_type):
    """Select the specimens for `cell_type` and extract their features."""
    specimen_ids = _select_specimen_ids(df, cell_type)
    print(f"{cell_type}: {len(specimen_ids)} specimens", flush=True)

    features = extract_ephys(ctc, specimen_ids)
    return pd.DataFrame.from_dict(features, orient="index")



# --------------------------------------------------------------------------
# main function (call from command line)
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cell-type",
        choices=sorted(CELL_TYPES),
        help="which cell type to extract",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="extract all cell types",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory where result should be stored",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    details_path = data_dir / SPECIMEN_DETAILS
    if not details_path.exists():
        parser.error(f"specimen details not found: {details_path}")

    df = pd.read_csv(details_path)

    # load data from cache to data_dir
    ctc = CellTypesCache(
        manifest_file=str(data_dir / "cell_types" / "manifest.json")
    )

    cell_types = sorted(CELL_TYPES) if args.all else [args.cell_type]

    for cell_type in cell_types:
        spec = CELL_TYPES[cell_type]
        result = extract_for_cell_type(ctc, df, cell_type)

        out_dir = data_dir / spec["subdir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{spec['stem']}_new"

        csv_path = out_dir / f"{stem}.csv"
        pkl_path = out_dir / f"{stem}.pkl"
        result.to_csv(csv_path)
        result.to_pickle(pkl_path)

        print(f"{cell_type}: wrote {len(result)} specimens to {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
