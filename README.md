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
| `requirements-allen.txt` | Allen Cell Types Database analysis — only the notebooks in `allen_data/` | **Separate** environment (see file header) |

Core packages: **NEURON 8.2.7**, **hnn-core**, numpy, scipy, pandas,
matplotlib, [cmcrameri](https://www.fabiocrameri.ch/colourmaps/) (colour maps),
[MNE-Python](https://mne.tools) (time-frequency analysis for beta events), and
JupyterLab.

### hnn-core version

The results in this manuscript were produced with a **development fork** of
hnn-core, not a released version:

- Fork: `https://github.com/katduecker/hnn-core`
- Branch: `pr-10`
- Commit: `195d280d`
- Reported version: `0.4.4.dev0`

`requirements.txt` pins this commit directly. **Before submission**, replace it
with the upstream commit or released version once the changes are merged, and
link the relevant pull requests here:

- TODO — hnn-core PRs described by this manuscript

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

TODO — what is included in the repository, what must be downloaded (Allen Cell
Types Database, MEG/ERP recordings, NHP laminar LFP), licences, and DOIs for
any separately deposited dataset.
Also mention the spectralevents data copied into the data folder and where it comes from

---

## Licence

TODO — add a `LICENSE` file. BSD-3-Clause is suggested, to match hnn-core.
Consider licensing data and figures separately under CC-BY-4.0.

## Citation

See [`CITATION.cff`](CITATION.cff) (currently a placeholder).
