"""optimize cells based on kalmbach_2021""" 

import os

pth = '/users/kduecker/hnn-tuning'
os.chdir(os.path.join(pth,'new_model_dev'))

from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

import numpy as np
from functools import partial
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.signal import windows
from scipy.signal import chirp
import pygad
import pickle

import multiprocessing

# own utils
from cell_builder_utils import l5_params, _get_dends, _get_pyr_soma, Cell
from hay_mechs_utils import Mechs


from hay_mechs_utils import Mechs, Ih_condu_apical, linear_gbar

def load_data(data_path):
    """load electrophysio data from humans"""
    # load electrophysio from Kalmbach, 2021
    df = pd.read_csv(data_path)

    df['etype'].value_counts() # 1 seems to be the E-type cells

    l5df = df[df['etype'] == 1]

    ephys_measures = dict()
    ephys_measures['target_rmp'] = np.mean(l5df['RMP'].values)

    ephys_measures['sag_ratio'] = np.mean(l5df['Sag_ratio'])

    ephys_measures['rebound'] = np.mean(l5df['Rebound'])

    ephys_measures['res_freq'] = np.mean(l5df['Fr'])

    ephys_measures['ap_thresh'] = np.mean(l5df['AP_threshold'])

    ephys_measures['ap_peak'] = np.mean(l5df['AP_peak'])

    ephys_measures['fi_slope'] = np.mean(l5df['FI_slope_max'])

    ephys_measures['ss_inp_res'] = np.mean(l5df['SSRn'])

    return ephys_measures

def return_start_params():
    """Hand-tuned starting values for the conductances"""

    return {'soma_gbar_NaTs2_t': 20_400e-4*10, 
                        'soma_gbar_SKv3_1': 6_930e-4,
                        'soma_gbar_Nap_Et2': 17.2e-4,
                        'soma_gbar_Ca_HVA': 9.92e-04,
                        'soma_gbar_Ca_LVAst': 34.3e-4,
                        'soma_gbar_SK_E2': 441e-4,
                        'soma_gbar_Ih': 1e-4,
                        'soma_gbar_Im': 0,
                        'soma_gbar_K_Pst': 22.3e-4,
                        'soma_gbar_K_Tst': 812e-4,
                        'soma_g_pas':  .338e-4,
                        'soma_e_pas': -90,
                        'soma_decay_CaDynamics_E2': 460,
                        'soma_gamma_CaDynamics_E2': .000501,
                        'basal_gbar_NaTs2_t': 20_400e-4*.1,
                        'basal_gbar_SKv3_1': 6_930e-4*0.1,
                        'basal_gbar_Ih': 5.14e-5,
                        'basal_g_pas': 1.75e-5,
                        'basal_e_pas':-90,
                        'apical_gbar_NaTa_t': 0.0213,
                        'apical_gbar_SKv3_1': 0.000261,
                        'apical_gbar_Ca_HVA': 2.78e-5,
                        'apical_gbar_Ca_LVAst': 93.5e-6,
                        'apical_gbar_SK_E2': 0.0012,
                        'apical_gbar_Ih': 1e-4,
                        'apical_gbar_Im': 0.0000675,
                        'apical_gbar_K_Pst': 0,
                        'apical_gbar_K_Tst': 0,
                        'apical_g_pas': 0.0000589*1.5,
                        'apical_e_pas': -85,
                        'apical_decay_CaDynamics_E2': 122,
                        'apical_gamma_CaDynamics_E2': .0005096*8}


def add_condu_cell(l5pyr, keys, conductances, hotzone=[1500,200]):
    """set conductances in each compartment of the cell"""

    # neuron imports
    import neuron
    from neuron import h
    from neuron.units import ms, mV

    basal_sections = ['basal_1', 'basal_2', 'basal_3', 'apical_oblique']
    apical_sections = ['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft']

    h.distance(sec = l5pyr._nrn_sections['soma'])

    for key, val in zip(keys, conductances):
        idx1 = key.find("_")
        idx2 = key.find("_", idx1+1)
        attr = key[idx1+1:]
        mech = key[idx2+1:]

        if key[:idx1] == 'soma':
            l5pyr._nrn_sections[key[:idx1]].insert(mech)
            for seg in l5pyr._nrn_sections['soma']:
                setattr(seg, attr, val)
        elif key[:idx1] == 'basal':
            for section in basal_sections:
                l5pyr._nrn_sections[section].insert(mech)
                for seg in l5pyr._nrn_sections[section]:
                    setattr(seg, attr, val)
        elif key[:idx1] == 'apical':
            for section in apical_sections:
                l5pyr._nrn_sections[section].insert(mech)
                for seg in l5pyr._nrn_sections[section]:
                    if mech == 'Ca_HVA':
                        distance = h.distance(seg.x, sec=seg.sec)
                        gbar = linear_gbar(val, val*7.5,1500, distance)
                        if (distance >= hotzone[0]) and (distance < hotzone[1]):
                                gbar *= 3
                    elif mech == 'Ca_LVAst':
                        distance = h.distance(seg.x, sec=seg.sec)
                        gbar = linear_gbar(val, val*3,1500, distance)
                        if (distance >= hotzone[0]) and (distance < hotzone[1]):
                                gbar *= 3
                    elif mech == 'Ih':
                        distance = h.distance(seg.x, sec=seg.sec)
                        gbar = Ih_condu_apical(val, distance)
                    else:
                        gbar = val
                    setattr(seg, attr, gbar)
    
    return l5pyr

# initialize neuron cell


"""In silico experiments"""

def rmp_grad(keys, conductances):
    """Resting membrane potential and gradient (penalize fast drops to RMP)"""

    # neuron imports
    import neuron
    from neuron import h
    from neuron.units import ms, mV

    pth_mod = os.path.join(pth,'local_hnn/hnn_core/mod')
    neuron.load_mechanisms(pth_mod)
    h.load_file("stdrun.hoc")


    # Get parameters
    start_params = return_start_params()
    hand_tuned_condu = np.array(list(start_params.values()))
    keys = list(start_params.keys())

    # Get morphological parameters
    section_params, end_pts, cell_tree = l5_params()
    sections = _get_dends(section_params, 'L5Pyr', list(end_pts.keys()), end_pts, v_init={'all': -60})
    sections['soma'] = _get_pyr_soma(section_params, 'L5Pyr', end_pts['soma'], v_init=-60)
    
    # Build cell template
    l5pyr = Cell(sections, cell_tree, cell_type='L5_pyramidal', gid=0)
    l5pyr.build()

    l5pyr = add_condu_cell(l5pyr, keys, conductances)

    try:
        t = h.Vector().record(h._ref_t)
        soma_voltg = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)

        soma_v = soma_voltg.as_numpy()
        h.celsius = 37
        h.finitialize()
        h.continuerun(100 * ms)
        rmp = soma_voltg[-3]
        grad = np.max(np.abs(np.gradient(soma_voltg)))

        if np.isnan(rmp):
            rmp = 1000
            grad = -1000

    except Exception as e:
        print(f"rmp_grad ERROR: {str(e)}", flush=True)
        return 1000, -1000, [], []
    
    del l5pyr

    return rmp, grad, soma_voltg, t

def rebound(keys, conductances):
    """Rebound slope and sag ratio"""

    # neuron imports
    import neuron
    from neuron import h
    from neuron.units import ms, mV

    pth_mod = os.path.join(pth,'local_hnn/hnn_core/mod')
    neuron.load_mechanisms(pth_mod)
    h.load_file("stdrun.hoc")

    # Get morphological parameters
    section_params, end_pts, cell_tree = l5_params()
    sections = _get_dends(section_params, 'L5Pyr', list(end_pts.keys()), end_pts, v_init={'all': -60})
    sections['soma'] = _get_pyr_soma(section_params, 'L5Pyr', end_pts['soma'], v_init=-60)
    
    ampl = np.linspace(-.1, -2, 10)

    soma_v = np.zeros((len(ampl),int(1/h.dt*205+1)))
    peak_heights = np.zeros_like(ampl)
    steady_state = np.zeros_like(ampl)
    min_membr_pot = np.zeros_like(ampl)
    sag_ratio = np.zeros_like(ampl)

    for i,amp in enumerate(ampl):

        # Build cell template
        l5pyr = Cell(sections, cell_tree, cell_type='L5_pyramidal', gid=0)
        l5pyr.build()
        l5pyr = add_condu_cell(l5pyr, keys, conductances)

        try:
            stim = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
            stim.delay = 5
            stim.dur = 100
            stim.amp = amp

            t = h.Vector().record(h._ref_t)
            soma_voltg = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)

            h.celsius = 37
            h.finitialize()
            h.continuerun(205 * ms)

            steady_idx = np.argmin(np.abs(t-90))
            soma_v[i] = soma_voltg.as_numpy()

            pk = find_peaks(soma_v[i][steady_idx:], height=[-70, 50])
            if len(pk[0]) > 1:
                peak_heights[i] = pk[1]['peak_heights'][0]
            elif len(pk[0]) == 0:
                peak_heights[i] = soma_v[i][steady_idx]
            else:
                peak_heights[i] = pk[1]['peak_heights'].item()
            
            min_membr_pot[i] = np.min(soma_v[i])

            steady_state[i] = soma_v[i][steady_idx]
        except Exception as e:
            print(f"rebound ERROR: {str(e)}", flush=True)
        finally:
            h('forall delete_section()')

        del l5pyr

    rebound_model = np.polyfit(steady_state, peak_heights,1)[0]
    inp_res_max = np.polyfit(ampl,min_membr_pot,1)[0]
    inp_res_steady = np.polyfit(ampl,steady_state,1)[0]

    sag_ratio_model = inp_res_max/inp_res_steady

    return rebound_model, sag_ratio_model, inp_res_max

def ap_measures(keys, conductances):
    """Action potential threshold and peak"""

    # neuron imports
    import neuron
    from neuron import h
    from neuron.units import ms, mV

    pth_mod = os.path.join(pth,'local_hnn/hnn_core/mod')
    neuron.load_mechanisms(pth_mod)
    h.load_file("stdrun.hoc")

    # Get morphological parameters
    section_params, end_pts, cell_tree = l5_params()
    sections = _get_dends(section_params, 'L5Pyr', list(end_pts.keys()), end_pts, v_init={'all': -60})
    sections['soma'] = _get_pyr_soma(section_params, 'L5Pyr', end_pts['soma'], v_init=-60)
    hotzone = [1500, 2000]
    try:
        # Build cell template
        l5pyr = Cell(sections, cell_tree, cell_type='L5_pyramidal', gid=0)
        l5pyr.build()
        l5pyr = add_condu_cell(l5pyr, keys, conductances)

        stim = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
        stim.delay = 20
        stim.dur = 3
        stim.amp = 3
        clamp_current = h.Vector().record(stim._ref_i)

        t = h.Vector().record(h._ref_t)
        v_soma = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)

        h.celsius = 37
        h.dt = .01
        h.finitialize()
        h.continuerun(50 * ms)

        ap, properties = find_peaks(v_soma.as_numpy(), height=0, width=0)

        v_grad = np.gradient(v_soma)
        ap_thresh_idx = np.argmin(np.abs(v_grad-np.max(v_grad)*.09))
        ap_thresh = v_soma[ap_thresh_idx]

        if len(ap) > 0:
            ap_peak = properties['peak_heights'][0]
        else:
            ap_peak = v_soma[-1]
        
    except Exception as e:
        print(f"AP ERROR: {str(e)}", flush=True)
    finally:
        h('forall delete_section()')

    del l5pyr
    return ap_thresh, ap_peak, v_soma, t


def fi_slope(keys, conductances):
    """F-I slope"""

    # neuron imports
    import neuron
    from neuron import h
    from neuron.units import ms, mV

    pth_mod = os.path.join(pth,'local_hnn/hnn_core/mod')
    neuron.load_mechanisms(pth_mod)
    h.load_file("stdrun.hoc")

    # Get morphological parameters
    section_params, end_pts, cell_tree = l5_params()
    sections = _get_dends(section_params, 'L5Pyr', list(end_pts.keys()), end_pts, v_init={'all': -60})
    sections['soma'] = _get_pyr_soma(section_params, 'L5Pyr', end_pts['soma'], v_init=-60)
    hotzone = [1500, 2000]

    ampls = np.linspace(.5,2,10)

    try:
        F = np.zeros_like(ampls)
        for i,amp in enumerate(ampls):

            # Build cell template
            l5pyr = Cell(sections, cell_tree, cell_type='L5_pyramidal', gid=0)
            l5pyr.build()
            l5pyr = add_condu_cell(l5pyr, keys, conductances)

            stim = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
            stim.delay = 1
            stim.dur = 1000
            stim.amp = amp

            t = h.Vector().record(h._ref_t)
            soma_voltg = h.Vector().record(l5pyr._nrn_sections['soma'](0.5)._ref_v)

            h.celsius = 37
            h.finitialize()
            h.continuerun(1001 * ms)

            soma_voltg = soma_voltg.as_numpy()

            spk, prop = signal.find_peaks(soma_voltg, height=0)

            F[i] = len(spk)

        FI_slope = np.diff(np.array(F))/np.diff(ampls)[0]
        pA = 0.001 # nA (convert nA to pA)

        fi = max(FI_slope)*pA 
    
    except Exception as e:
        print(f"FI ERROR: {str(e)}", flush=True)
    finally:
        h('forall delete_section()')

    return fi

def res_to_chirp(sections, cell_tree, keys, conductances):
    """Resonance frequency to chirp stimulus"""

    # neuron imports
    import neuron
    from neuron import h
    from neuron.units import ms, mV

    l5pyr= Cell(sections, cell_tree, cell_type='L5_pyramidal', gid=0)
    l5pyr.build()
    l5pyr = add_condu_cell(l5pyr, keys, conductances)

    # Define simulation parameters
    h.tstop = 20_100  # Simulation duration in ms
    h.dt = 0.1    # Time step

    # Create a chirp stimulus (linear frequency sweep)
    t_chirp = np.arange(0, h.tstop, h.dt)  # Time vector

    # Kalmbach et al: linear increase from 1-15 Hz over
    f0 = 0.2  # Initial frequency in Hz
    f1 = 40  # Final frequency in Hz
    A = .2  # Amplitude of the chirp in nA - adjusted to achieve a voltage deflection of ~10mV peak to peak 

    # logarithmic chirp
    chirp_signal = A * chirp(t_chirp/1000, f0, 20, f1, method='log') + 0.2

    # set up clamp
    stim = h.IClamp(l5pyr._nrn_sections['soma'](0.5))
    stim.delay = 0
    stim.dur = h.tstop+stim.delay
    stim.amp = 0  # We'll use Vector.play() to set the amplitude
    # Apply the chirp stimulus using a Vector and play()
    stim_vec = h.Vector(chirp_signal)
    t = np.arange(0, h.tstop, h.dt)
    time_vec = h.Vector(t)
    stim_vec.play(stim._ref_amp, time_vec, 1)

    # Record membrane potential
    v_rec = h.Vector()
    v_rec.record(l5pyr._nrn_sections['soma'](0.5)._ref_v)
    t_rec = h.Vector().record(h._ref_t)
    # Run the simulation
    h.finitialize()  # Initialize membrane potential
    h.continuerun(h.tstop)
    h.dt = 0.1

    gauss_win = windows.gaussian(len(chirp_signal), len(chirp_signal)/1.5)

    N = len(v_rec)
    T = h.dt/1000
    freq = fftfreq(N, T)[:N//2]
    lowf = int(np.argmin(np.abs(freq-0.2)))

    fft_membr = fft(v_rec.as_numpy()[1:])
    fft_chirp = fft(chirp_signal)

    impedance = fft_membr/fft_chirp
    zap = np.abs(impedance)
    zap = zap[:N//2]

    zap_smooth = np.zeros(zap.shape[0]-10)

    for c in range((len(zap)-10)):  
        zap_smooth[c] = np.mean(zap[c:c+10])

    res_freq = freq[np.argmax(zap_smooth[lowf:])-lowf]

    return res_freq, chirp_signal, v_rec, t_rec, zap

def fitness_function(ga_instance, solution, solution_idx):

    # neuron imports
    import neuron
    from neuron import h
    from neuron.units import ms, mV

    ephys_measures = load_data('/users/kduecker/hnn-tuning/data/kalmbach_2021_l5_electrophys.csv')

    # Get parameters
    start_params = return_start_params()
    hand_tuned_condu = np.array(list(start_params.values()))
    keys = list(start_params.keys())

    conductances = hand_tuned_condu*solution

    # # Fitness
    fitness = np.zeros(2,)

    print("Testing RMP...", flush=True)
    rmp, grad, soma_voltg, t = rmp_grad(keys, conductances)

    # RMP
    fitness[0] = np.mean(np.array((1/(np.abs(rmp-ephys_measures['target_rmp'])+.01),1/(abs(0.05-grad)+0.01))));
    
    print("RMP done. Testing AP...", flush=True)
    # AP measures
    ap_thresh, ap_peak, v_soma, t = ap_measures(keys, conductances)

    ap_thresh_fit = 1/(np.abs(ap_thresh-ephys_measures['ap_thresh'])+0.01)
    ap_peak_fit = 1/(np.abs(ap_peak-ephys_measures['ap_peak'])+0.01)

    fitness[1] = np.mean(np.array((ap_thresh_fit, ap_peak_fit)))

    print("AP done. Testing FI...", flush=True)
    # # FI slope
    # fi_slo = fi_slope(keys, conductances)

    # fitness[2] = 1/(np.abs(fi_slo-ephys_measures['fi_slope'])+0.01)

    # # Rebound

    # rebound_model, sag_ratio_model, inp_res_max = rebound(sections, cell_tree, keys, conductances)

    # inp_res = 1/(abs(inp_res_max-ephys_measures['ss_inp_res'])+0.01)
    
    # reb = 1/(np.abs(rebound_model-ephys_measures['rebound'])+0.01)

    # sagr = 1/(np.abs(sag_ratio_model-ephys_measures['sag_ratio'])+0.01)

    # fitness[3] = (inp_res + reb + sagr)/3

    return fitness


def fitness_function_wrapper(ga_instance, solution, solution_idx):

    from func_timeout import func_timeout, FunctionTimedOut

    try:
        # Timeout after 300 seconds (5 minutes) per fitness evaluation
        return func_timeout(300, fitness_function, args=(ga_instance, solution, solution_idx))
    except FunctionTimedOut:
        print(f"Fitness evaluation timed out for solution {solution_idx}. Penalizing.", flush=True)
        return -np.inf  # Assign worst possible fitness

def on_fitness(ga_instance, population_fitness):
    print("on_fitness()", flush=True)

def on_parents(ga_instance, selected_parents):
    print("on_parents()", flush=True)

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()", flush=True)

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()", flush=True)

def on_generation(ga_instance):
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        print(f'Generation : {ga_instance.generations_completed} at {now.date()} {current_time}', flush=True)

        np.save(f'{pth}/new_model_dev/l5_last_gen_parents_rmp_ap.npy', ga_instance.last_generation_parents)
        np.save(f'{pth}/new_model_dev/l5_current_pop_rmp_ap.npy', ga_instance.population)
        np.save(f'{pth}/new_model_dev/l5_last_gen_parents_fit_rmp_ap.npy',
                ga_instance.last_generation_fitness[ga_instance.last_generation_parents_indices])
        fitness_list.append(ga_instance.last_generation_fitness)
        np.save(f'{pth}/new_model_dev/l5_generation_fitness_rmp_ap.npy', np.array(fitness_list))
        parents_list.append(ga_instance.last_generation_parents)
        np.save(f'{pth}/new_model_dev/l5_all_parents_rmp_ap.npy', np.array(parents_list))

        ga_instance.save('l5_pygad_inst_rmp_ap')

def on_stop(ga_instance, last_population_fitness):
    
    print("Done")
    best_sol = ga_instance.best_solution()

    with open('l5_best_sol_rmp_ap.pkl', 'wb') as file:
        pickle.dump(best_sol, file)

    # # Loading the tuple
    # with open('l5_best_sol.pkl', 'rb') as file:
    #     loaded_tuple = pickle.load(file)

def main():
    
    num_gen = 150
    num_parents_mating = 256
    population_size = 512
    
    gene_space = [[.01, 15],
                [0.01, 15],
                [0.01, 15],
                [0.01, 15],
                [0.01, 15],
                [0.01, 15], 
                [0.01, 15],
                [1, 1],
                [0.01, 15],
                [0.01, 15],
                [0.01, 15],
                [0.8, 1.2],
                [0.4, 1.2],
                [0.01, 15],
                [0.01, 15],
                [0.01, 15],
                [0.01, 15], 
                [0.01, 15],
                [0.7, 1.2],
                [0.01, 15], 
                [0.01, 15],
                [0.01, 15],
                [0.01, 15],
                [0.01, 15],
                [0.01, 15],
                [0.01, 15],
                [1, 1],
                [1, 1],
                [0.01, 15],
                [0.8, 1.2],
                [0.01, 4],
                [0.01, 15]]

    initial_population = np.ones((len(gene_space),))
    initial_population_2d = np.zeros((population_size, 32))

    for i,gs in enumerate(gene_space):
        initial_population_2d[:,i] = np.random.uniform(low=gs[0], high=gs[1],size=(population_size,))

    initial_population_2d[0] = initial_population  

    global fitness_list
    global parents_list

    fitness_list = []
    parents_list = []

    ga_instance = pygad.GA(num_generations=num_gen,
                    num_parents_mating=num_parents_mating,
                    keep_parents = -1,
                    fitness_func=fitness_function_wrapper,
                    num_genes=len(gene_space),
                    initial_population=initial_population_2d,
                    gene_space=gene_space,
                    on_fitness=on_fitness,
                    on_parents=on_parents,
                    on_crossover=on_crossover,
                    on_mutation=on_mutation,
                    on_generation=on_generation,
                    on_stop=on_stop,
                
                    parallel_processing=["process", 63])

    ga_instance.run()

    ga_instance.save('l5_pygad_inst')

if __name__ == "__main__":
    main()