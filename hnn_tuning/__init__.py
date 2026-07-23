"""Helper code shared across the hnn-tuning analyses.

Importable as `hnn_tuning` once the repository is installed with

    pip install -e .

The modules here are currently named by kind (``*_mechs_utils``,
``*_plot_utils``); consolidation into topic modules (``cells``,
``mechanisms``, ``plotting``, ``connectivity``) is in progress.

Repository locations are re-exported here so scripts and notebooks can obtain
them without hardcoding absolute paths::

    from hnn_tuning import DATA, SINGLE_CELL_SIM, HNN_CORE_MOD
"""

from ._paths import (
    REPO_ROOT,
    SINGLE_CELL_SIM,
    NETWORK_SIM,
    DATA,
    RESULTS,
    HNN_CORE_MOD,
    ALLEN_DATA,
    results_dir,
)

__all__ = [
    "REPO_ROOT",
    "SINGLE_CELL_SIM",
    "NETWORK_SIM",
    "DATA",
    "RESULTS",
    "HNN_CORE_MOD",
    "ALLEN_DATA",
    "results_dir",
]
