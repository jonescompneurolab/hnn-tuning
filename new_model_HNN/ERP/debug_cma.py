import numpy as np
import matplotlib.pyplot as plt
from hnn_core import jones_2009_model, simulate_dipole, JoblibBackend
from hnn_core.network_models import add_erp_drives_to_jones_model
from hnn_core.optimization import Optimizer, add_opt_drives, set_params_opt_drives
from hnn_core.dipole import _corr, _rmse, average_dipoles
from hnn_core import MPIBackend
n_procs=20

# find MPI
import sys
import os
import subprocess

# Use conda's newer libstdc++ instead of the system one
conda_lib = os.path.expanduser("~/.conda/envs/hnn-core-env/lib")
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_path}"

# UCX settings (from before)
os.environ["UCX_TLS"] = "tcp,self,sm"
os.environ["UCX_NET_DEVICES"] = "all"

# MPI paths
mpi_bin = "/oscar/rt/sw/external/hpcx-mpi/2.25.1/bin"
mpi_lib = "/oscar/rt/sw/external/hpcx-mpi/2.25.1/lib"
if mpi_bin not in os.environ["PATH"]:
    os.environ["PATH"] = f"{mpi_bin}:{os.environ['PATH']}"
if mpi_lib not in ld_path:
    os.environ["LD_LIBRARY_PATH"] = f"{mpi_lib}:{os.environ['LD_LIBRARY_PATH']}"
    
# Add MPI bin directory to PATH
mpi_bin = "/oscar/rt/sw/external/hpcx-mpi/2.25.1/bin"
if mpi_bin not in os.environ["PATH"]:
    os.environ["PATH"] = f"{mpi_bin}:{os.environ['PATH']}"

print("Updated PATH to include MPI")
print("MPI location:", mpi_bin)

print("PATH:", os.environ.get('PATH', 'NOT SET'))
print("Which mpiexec:")
result = subprocess.run(['which', 'mpiexec'], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "mpiexec not found")

tstop = 200
dt = 0.5  # Just used for testing, 0.025 is the default but will be slower
scaling_factor = 3000
smooth_win = 30

# Create target dipole
net_target = jones_2009_model()
add_erp_drives_to_jones_model(net_target)
with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'):
        target_dpl = simulate_dipole(net_target, tstop=tstop, dt=dt, verbose=False)
target_dpl = target_dpl[0].copy().smooth(smooth_win).scale(scaling_factor)

# Create base network with drives to be optimized
net_base = jones_2009_model()
net_base._verbose = False
constraints, initial_params = add_opt_drives(net_base, n_prox=2, n_dist=1)

# Run optimization
max_iter = 10
optim = Optimizer(net_base, tstop=tstop, constraints=constraints, solver='cma',
                set_params=set_params_opt_drives, initial_params=initial_params, max_iter=max_iter, obj_fun="dipole_corr")
                
popsize = 5 
with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'): 
        optim.fit(target=target_dpl, n_trials=5, scale_factor=scaling_factor,
                smooth_window_len=smooth_win, dt=dt, popsize=popsize)
with MPIBackend(n_procs=n_procs, mpi_cmd='mpiexec'): 
    opt_dpl = simulate_dipole(optim.net_, tstop=tstop, verbose=False)

opt_dpl = [dpl.smooth(smooth_win).scale(scaling_factor) for dpl in opt_dpl]
avg_dpl = average_dipoles(opt_dpl)
obj = _corr(avg_dpl, target_dpl)
print(f'obj_ == corr:'{optim.obj_ == obj})