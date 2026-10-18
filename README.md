# hnn-tuning

Code, tutorials, and analysis accompanying the manuscript:

> **TODO — manuscript title, authors, year, journal/preprint DOI**

This repository contains the biophysical cell models, network configurations,
and analysis notebooks used to develop and validate the new `duecker_ET_model` developed for
[hnn-core](https://github.com/jonescompneurolab/hnn-core).

---

## Installation

> **Note (pre-release):** the `duecker_ET_model` is on hnn-core `master` but
> not yet in a hnn-core release. Until it is, `requirements.txt` installs
> hnn-core directly from GitHub at a pinned commit, so you need `git`
> available. Nothing else changes once the release is out.

Tested on macOS/arm64 with Python 3.12 and conda (Anaconda or Miniconda).

1. If you installed NEURON with the standalone installer package, uninstall it
   first so it does not shadow the pip-installed version.
2. macOS only: install the Xcode Command-Line Tools (restart afterwards):
   ```bash
   xcode-select --install
   ```
3. Create and activate the environment, and install MPI (needed for
   `MPIBackend`, used by the network simulations):
   ```bash
   conda create -y -n hnn-tuning python=3.12
   conda activate hnn-tuning
   conda install -y "openmpi>5" mpi4py -c conda-forge
   ```
4. macOS only: let NEURON find the conda MPI libraries whenever the
   environment is activated:
   ```bash
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
   echo "export OLD_DYLD_FALLBACK_LIBRARY_PATH=\$DYLD_FALLBACK_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   echo "export DYLD_FALLBACK_LIBRARY_PATH=\$DYLD_FALLBACK_LIBRARY_PATH:\${CONDA_PREFIX}/lib" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   echo "export DYLD_FALLBACK_LIBRARY_PATH=\$OLD_DYLD_FALLBACK_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
   echo "unset OLD_DYLD_FALLBACK_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
   conda deactivate && conda activate hnn-tuning
   ```
   For other operating systems see the
   [hnn-core installation guide](https://jonescompneurolab.github.io/hnn-core/stable/install.html).
5. Clone this repository and install the requirements plus the `hnn_tuning`
   helper package:
   ```bash
   git clone <repository-url>
   cd hnn-tuning
   pip install -r requirements.txt
   pip install -e .          # makes `import hnn_tuning` work
   ```
6. Check the install:
   ```bash
   python -c "import hnn_core, hnn_tuning; from hnn_core import MPIBackend; print(hnn_core.__version__)"
   ```

To work on hnn-core itself, clone it and run `pip install -e ".[dev]"`
inside the hnn-core directory **after** step 5. This replaces the pinned
version.

---

## Requirements

All analysis code is Python. Dependencies are split across two files because
they cannot coexist in one environment.

| File | Purpose | Install into |
|------|---------|--------------|
| `requirements.txt` | Single cell and network simulations, reproduces all figures in manuscript. | Main environment (Python 3.12) |
| `requirements-allen.txt` | Allen Cell Types Database analysis — only extraction of electrophysiological features in `extract_ephys/allen_get_ephys_data.py` | See **Data**. |

Core packages: **NEURON 8.2.7**, **hnn-core** (with **OpenMPI + mpi4py** for
`MPIBackend`, installed via conda), numpy, scipy, pandas, matplotlib,
[MNE-Python](https://mne.tools) (time-frequency analysis for beta events), and
JupyterLab.


---

## Repository layout

| Folder | Description | 
|------|---------|

hnn_tuning/ | Helper functions for simulations and plotting |
data/ | Pre-processed single cell data (can be replicated using extract_ephys/allen_get_ephys_data.py, see requirements-allen.txt | 
single_cell_sim | Simulations of single cell dynamics and comparison to data from Allen Cell Types database. |
network_sim | Simulations using the tuned network.|


---

## Data

All data required to produce the figures in the manuscript are shared in `data/`.
For the [Allen Cell Types data](https://celltypes.brain-map.org/), the extracted electrophysiological features
for the different cells (interneuron, l5, l23) are shared in `data/allen-data`.
These features were extracted using `extract_ephys/allen_get_ephys_data.py` which requires allensdk
and has to be run in a *separate* environment (see `requirements-allen.txt`).

---

## Citation

If you use this model for your own work, please cite as: 

[`CITATION.cff`](CITATION.cff) (TO-DO!!)
