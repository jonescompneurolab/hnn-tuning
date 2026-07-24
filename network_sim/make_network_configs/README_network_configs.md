# Building the network configuration files

**Only included for documentation (don't run).** The network configuration files ship with the
hnn-core fork, and every notebook in `network_sim/` loads them directly via
`read_network_configuration`.

## Usage

Run from this directory:

```bash
python make_network_configs.py --list      # show the registry: params file + drives per config
python make_network_configs.py --all       # rebuild everything
python make_network_configs.py --name duecker_ET_gamma_50Hz_bursty   # rebuild one
```

Add `--overwrite` to replace configuration files that already exist (by default
existing files are skipped).

Output goes to `hnn_core/network_configs/` inside the installed hnn-core.

## Output files

| Configuration | Source params | Drives |
| --- | --- | --- |
| `duecker_ET_gamma_50Hz_sustained` | `duecker_ET_gamma_50Hz_sustained.json` | Poisson, rate 200 Hz |
| `duecker_ET_gamma_50Hz_bursty` | `duecker_ET_gamma_50Hz_bursty.json` | Poisson, rate 100 Hz, reduced NMDA |
| `duecker_ET_alpha_beta_complex` | default `duecker_ET_model()` | proximal + distal bursty drives |
| `duecker_ET_default` | default `duecker_ET_model()` | none |

