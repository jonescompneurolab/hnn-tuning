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
   helper package. The repository is currently **private**, so you need to be
   a collaborator and authenticate with GitHub. Use one of:

   - **SSH** (recommended if you have an
     [SSH key added to GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)):
     ```bash
     git clone git@github.com:jonescompneurolab/hnn-tuning.git
     ```
   - **HTTPS**: GitHub no longer accepts your account password here.
     Authenticate first with `gh auth login` (GitHub CLI), or enter a
     [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
     when git asks for a password:
     ```bash
     git clone https://github.com/jonescompneurolab/hnn-tuning.git
     ```

   Then:
   ```bash
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
|--------|-------------|
| `hnn_tuning/` | Helper functions for simulations and plotting |
| `data/` | Pre-processed single-cell data (can be replicated using `extract_ephys/allen_get_ephys_data.py`, see **Data**) |
| `single_cell_sim/` | Simulations of single-cell dynamics and comparison to data from the Allen Cell Types Database |
| `network_sim/` | Simulations using the tuned network |
| `results/` | Saved simulation outputs used by the figure notebooks |

---

## Workflow for reproducing manuscript figures


| Simulation | Figure | Notebook / script to plot | Generates | 
|--------|--------|---------------------------|----------------------------|
| L5 ET pyramidal| Fig. 2 & 3; Supp. Fig. 1, Supp. Fig. 2 | single_cell_sim/cells/l5_pyramidal.ipynb | L5 ET pyramidal comparison to Allen data & calcium spikes, ion channel conductances, somato-dendritic attenuation, single neuron dipoles | 
| L2/3 Pyramidal | Fig. 4 & 5; Supp. Fig. 1, Supp. Fig. 2 | single_cell_sim/cells/l23_pyramidal.ipynb | L2/3 pyramidal comparison to Allen data & calcium spikes, ion channel conductances,  somato-dendritic attenuation, single neuron dipoles | 
|Interneuron | Fig. 6 | single_cell_sim/cells/interneuron.ipynb | Interneuron comparison to Allen data | 
| Synapses | Fig. 7 | network_sim/inspect_synapses/synapses.ipynb | NMDA synapse, slow GABAA, GABAB |
| Connectivity | Fig. 8 | network_sim/connectivity/connectivity.ipynb | Space constant, maximum weights, connectivity heat maps, PSPs |
| ERP | - | network_sim/ERP/step1_handtune.ipynb | Hand-tuned starting values not shown in manuscript |
| ERP | - | network_sim/ERP/step2_optimize_ERP_cma.py | Optimization script to fit ERP drives (requires HPC, results shared in results/network_sim/ERP) |
| ERP | Fig. 9 | network_sim/ERP/step3_inspect_ERP_multiscale.ipynb | Multiscale interpretation of ERP simulation in terms of spiking, firing rates, membrane voltage, Ca2+, synaptic currents |
| Alpha-Beta | - | network_sim/alpha-beta/step1_alpha_beta_complex_simulation.py | Simulates and analyses alpha-beta complex dipoles (HPC recommended, results shared in results/network_sim/alpha_beta) | 
| Alpha-Beta | Fig. 10 | network_sim/alpha-beta/step2_alpha_beta_complex_results.ipynb | Plots results of alpha-beta simulation, analyses SI MEG dipole as in Jones et al., 2009 |
| Gamma | - | network_sim/gamma/step1_gamma_burst_simulation.py | Simulates 100 trials of 1s each with bursty gamma configuration, results saved in results/network_sim/gamma (HPC recommended, not shared because too large) |
| Gamma | Fig. 11 | network_sim/gamma/step2_inspect_gamma_multiscale.ipynb | Plots single trial continuous gamma, and bursty gamma |
| comparison neymotin_2020_model | Supp. Fig. 3 | network_sim/comparison_neymotin2020/suppfig3.ipynb | Plots weights, space constants, and single cell dipoles for neymotin_2020_model, identifies rheobase for neymotin_2020_model and duecker_ET_model | 
| Supporting information/development | - | network_sim/dipole_baseline_correction/fit_baseline_correction.ipynb | Shows development & result of baseline correction. Included for documentation. | 
| Supporting information/development | - | network_sim/make_network_configs/make_network_configs.py | Generates network configurations from legacy "flat" json files and stores them in network_sim/network_configs. Included for documentation (not usable without flat json files). | 
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
