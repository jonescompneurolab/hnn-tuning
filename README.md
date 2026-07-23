# hnn-tuning

Code, tutorials, and analysis accompanying the manuscript:

> **TODO — manuscript title, authors, year, journal/preprint DOI**

This repository contains the biophysical cell models, network configurations,
and analysis notebooks used to develop and validate the updated pyramidal-cell
and network models in [hnn-core](https://github.com/jonescompneurolab/hnn-core).

---

## Quick start

```bash
git clone <repository-url>
cd hnn-tuning

conda create -n hnn-tuning python=3.12
conda activate hnn-tuning
pip install -r requirements.txt
pip install -e .          # installs the `hnn_tuning` helper package
```

KD note to self: make sure the right version of hnn-core is installed

---

## Requirements

All analysis code is Python. Dependencies are split across two files because
they cannot coexist in one environment.

| File | Purpose | Install into |
|------|---------|--------------|
| `requirements.txt` | Everything needed for the cell models, network simulations, and figures | Main environment (Python 3.12) |
| `requirements-allen.txt` | Allen Cell Types Database analysis — only extraction of electrophysiological features in `extract_ephys/allen_get_ephys_data.py` | See **Data**. |

Core packages: **NEURON 8.2.7**, **hnn-core**, numpy, scipy, pandas,
matplotlib, [cmcrameri](https://www.fabiocrameri.ch/colourmaps/) (colour maps),
[MNE-Python](https://mne.tools) (time-frequency analysis for beta events), and
JupyterLab.

### hnn-core version

TO-DO

---

## Repository layout

TODO

---

## Reproducing the figures

TODO — a table mapping each manuscript figure to the script that produces it,
with approximate runtimes.

| Figure | Script / notebook | Approx. runtime | Notes |
|--------|-------------------|-----------------|-------|
| | | | |

---

## Data

All data required to produce the figures in the manuscript are shared in `data/`.
For the [Allen Cell Types data](https://celltypes.brain-map.org/), the extracted electrophysiological features
for the different cells (interneuron, l5, l23) are shared in `data/allen-data`.
These features were extracted using `extract_ephys/allen_get_ephys_data.py` which requires allensdk
and have to be run in a *separate* environment (see `requirements_allen.txt`).

---

## Licence

TODO — add a `LICENSE` file. BSD-3-Clause is suggested, to match hnn-core.
Consider licensing data and figures separately under CC-BY-4.0.

## Citation

See [`CITATION.cff`](CITATION.cff) (currently a placeholder).
