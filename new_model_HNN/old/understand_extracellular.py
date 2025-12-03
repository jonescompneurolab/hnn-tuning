""""Understand extracellular mechanism in NEURON"""

from hnn_core import duecker_2025_ET_model, simulate_dipole

net = duecker_2025_ET_model(mesh_shape=(1,1))
depths = list(list(range(-200,2200,100)))
electrode_pos = [(2, 2, dep) for dep in depths]
net.add_electrode_array('probe1', electrode_pos, method='lsa') 
dpl = simulate_dipole(net, tstop=100.0, bsl_cor='calcium')