"""Filesystem locations used by the helper modules.

Everything is resolved relative to this file, so the package works from any
checkout without hardcoded, machine-specific paths.
"""

from pathlib import Path
import hnn_core
# hnn_tuning/_paths.py  ->  hnn_tuning/  ->  repository root
REPO_ROOT = Path(__file__).resolve().parents[1]

SINGLE_CELL_SIM = REPO_ROOT / "single_cell_sim"
NETWORK_SIM = REPO_ROOT / "network_sim"
DATA = REPO_ROOT / "data"


# Generated figures and other outputs, organised by stage and cell type,
# e.g. RESULTS / "single_cell_sim" / "interneuron". Regenerable — not tracked.
RESULTS = REPO_ROOT / "results"

# Allen Cell Types electrophysiology (interneuron / l23 / l5 subdirectories).
ALLEN_DATA = DATA / "allen-data"


def results_dir(*parts):
    """Return (and create) a results subdirectory, e.g.
    results_dir("single_cell_sim", "interneuron")."""
    d = RESULTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d

# HNN CORE ROOT

hnn_core_root = Path(hnn_core.__file__).parent
HNN_CORE_MOD = hnn_core_root / "mod"

# Directory holding the custom epsp.mod point process (used by attenuation
# experiments in single_cell_sim).
EPSP_MOD = SINGLE_CELL_SIM
