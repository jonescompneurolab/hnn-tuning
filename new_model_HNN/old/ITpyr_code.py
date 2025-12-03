def get_L5PyrIT_params():

    return {
        # Soma
        'L5Pyr_soma_L': 39.,
        'L5Pyr_soma_diam': 28.9,
        'L5Pyr_soma_cm': 1,
        'L5Pyr_soma_Ra': 100, #Rich 495.73, Hay: 100

        # Dendrite
        'L5Pyr_dend_cm': 1,
        'L5Pyr_dend_Ra': 100, #Rich 495.73, Hay: 100

        'L5Pyr_apicaltrunk_L': 102.,
        'L5Pyr_apicaltrunk_diam': 10.2,

        'L5Pyr_apical1_L': 680.,
        'L5Pyr_apical1_diam': 7.48,

        'L5Pyr_apical2_L': 680.,
        'L5Pyr_apical2_diam': 4.93,

        'L5Pyr_apicaltuft_L': 425.,
        'L5Pyr_apicaltuft_diam': 3.4*0.5,

        'L5Pyr_apicaloblique_L': 255.,
        'L5Pyr_apicaloblique_diam': 5.1,

        'L5Pyr_basal1_L': 85.,
        'L5Pyr_basal1_diam': 6.8,

        'L5Pyr_basal2_L': 255.,
        'L5Pyr_basal2_diam': 8.5,

        'L5Pyr_basal3_L': 255.,
        'L5Pyr_basal3_diam': 8.5,

        # Synapses
        'L5Pyr_ampa_e': 0.,
        'L5Pyr_ampa_tau1': 0.5,
        'L5Pyr_ampa_tau2': 5.,

        'L5Pyr_nmda_e': 0.,
        'L5Pyr_nmda_tau1': 1.,
        'L5Pyr_nmda_tau2': 20.,

        'L5Pyr_gabaa_e': -80.,
        'L5Pyr_gabaa_tau1': 0.5,
        'L5Pyr_gabaa_tau2': 5.,

        'L5Pyr_gabab_e': -80.,
        'L5Pyr_gabab_tau1': 45,
        'L5Pyr_gabab_tau2': 200.,

        # Biophysics soma
        'L5Pyr_soma_gbar_NaTs2_t_32d': 20_400e-4/2,
        'L5Pyr_soma_gbar_SKv3_1': 6_930e-4/2,
        'L5Pyr_soma_gbar_Nap_Et2': 17.2e-4/2*0.5,
        'L5Pyr_soma_gbar_SK_E2': (441e-4)*0.01,
        'L5Pyr_soma_gbar_Ca_HVA': 9.92e-04/2*0.1,
        'L5Pyr_soma_gbar_Ca_LVAst': 34.3e-4/2*0.5,
        'L5Pyr_soma_gbar_Ih': 1e-4/2*.55,
        'L5Pyr_soma_gbar_Im': 1e-4*50,
        'L5Pyr_soma_g_pas': 1.0 / 12000.0*0.1,
        'L5Pyr_soma_e_pas': -65,
        'L5Pyr_soma_decay_CaDynamics_E2' : 460,
        'L5Pyr_soma_gamma_CaDynamics_E2' : .000501,

        # Biophysics basal
        'L5Pyr_basal_gbar_NaTs2_t_32d': 20_400e-4/2,
        'L5Pyr_basal_gbar_SKv3_1': 0.000261/2,
        'L5Pyr_basal_gbar_Ih': 5.14e-5/2*0.4,
        'L5Pyr_basal_g_pas': 1.0 / 12000.0*0.1,
        'L5Pyr_basal_e_pas': -65,

        # Biophysics dends
        'L5Pyr_dend_gbar_NaTa_t_32d': 0.0213/2*1.5,
        'L5Pyr_dend_gbar_SKv3_1': 0.000261/2,
        'L5Pyr_dend_gbar_SK_E2': 0.0012/2,
        'L5Pyr_dend_gbar_Ca_HVA': 2.78e-5/2*2,
        'L5Pyr_dend_gbar_Ca_LVAst': 93.5e-6/2*2,
        'L5Pyr_dend_gbar_Ih': 1e-4/2*0.55,
        'L5Pyr_dend_gbar_Im': 0.0000675/2*50,
        'L5Pyr_dend_g_pas':  1.0 / 12000.0*0.1,
        'L5Pyr_dend_e_pas': -65,
        'L5Pyr_dend_decay_CaDynamics_E2' : 122,
        'L5Pyr_dend_gamma_CaDynamics_E2' : .0005096*.8
    }


def human_ETIT_model(params=None, add_drives_from_params=False,
                  legacy_mode=False, mesh_shape=(10, 10), et_it_ratio=30):


    """"Initiate like old calcium model and then replace with new cells"""

    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    if params is None:
        params = read_params(params_fname)

    net = Network(params, add_drives_from_params=add_drives_from_params,
                  legacy_mode=legacy_mode, mesh_shape=mesh_shape)
    

    delay = net.delay

    # layer2 Pyr -> layer2 Pyr
    # layer5 Pyr -> layer5 Pyr
    lamtha = 3.0
    loc = 'proximal'
    target_cell ='L2_pyramidal'
    for receptor in ['nmda', 'ampa']:
        key = f'gbar_{_short_name(target_cell)}_'\
                f'{_short_name(target_cell)}_{receptor}'
        weight = net._params[key]*2
        net.add_connection(
            target_cell, target_cell, loc, receptor, weight,
            delay, lamtha, allow_autapses=False)
    
    target_cell ='L5_pyramidal'
    for receptor in ['nmda', 'ampa']:
        key = f'gbar_{_short_name(target_cell)}_'\
                f'{_short_name(target_cell)}_{receptor}'
        weight = net._params[key]*2

        net.add_connection(
            target_cell, target_cell, loc, receptor, weight,
            delay, lamtha, allow_autapses=False)
    

    # reduce the gabab connection slightly because it has a longer time constant
    # layer2 Basket -> layer2 Pyr
    src_cell = 'L2_basket'
    target_cell = 'L2_pyramidal'
    lamtha = 50.
    loc = 'soma'
    receptor='gabaa'
    key = f'gbar_L2Basket_L2Pyr_{receptor}'
    weight = net._params[key]
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)
    
    receptor='gabab'
    key = f'gbar_L2Basket_L2Pyr_{receptor}'
    weight = net._params[key]*0.5
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)
        

    # layer5 Basket -> layer5 Pyr
    src_cell = 'L5_basket'
    target_cell = 'L5_pyramidal'
    lamtha = 70.
    loc = 'soma'
    receptor = 'gabaa'
    key = f'gbar_L5Basket_{_short_name(target_cell)}_{receptor}'
    weight = net._params[key]
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)
    receptor = 'gabab'
    key = f'gbar_L5Basket_{_short_name(target_cell)}_{receptor}'
    weight = net._params[key]*0.5
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer2 Pyr -> layer5 Pyr
    src_cell = 'L2_pyramidal'
    lamtha = 3.
    receptor = 'ampa'
    for loc in ['proximal', 'distal']:
        key = f'gbar_L2Pyr_{_short_name(target_cell)}'
        weight = net._params[key]*4.5
        net.add_connection(
            src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer2 Basket -> layer5 Pyr
    src_cell = 'L2_basket'
    lamtha = 50.
    key = f'gbar_L2Basket_{_short_name(target_cell)}'
    weight = net._params[key]*2
    loc = 'distal'
    receptor = 'gabaa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # xx -> layer2 Basket
    src_cell = 'L2_pyramidal'
    target_cell = 'L2_basket'
    lamtha = 3.
    key = f'gbar_L2Pyr_{_short_name(target_cell)}'
    weight = net._params[key]/2
    loc = 'soma'
    receptor = 'ampa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    src_cell = 'L2_basket'
    lamtha = 20.
    key = f'gbar_L2Basket_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'soma'
    receptor = 'gabaa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # xx -> layer5 Basket
    src_cell = 'L5_basket'
    target_cell = 'L5_basket'
    lamtha = 20.
    loc = 'soma'
    receptor = 'gabaa'
    key = f'gbar_L5Basket_{_short_name(target_cell)}'
    weight = net._params[key]
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha,
        allow_autapses=False)

    src_cell = 'L5_pyramidal'
    lamtha = 3.
    key = f'gbar_L5Pyr_{_short_name(target_cell)}'
    weight = net._params[key]/3
    loc = 'soma'
    receptor = 'ampa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    src_cell = 'L2_pyramidal'
    lamtha = 3.
    key = f'gbar_L2Pyr_{_short_name(target_cell)}'
    weight = net._params[key]/3
    loc = 'soma'
    receptor = 'ampa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)
    
    
    # Replace cells

    # L5
    cell_name = 'L5_ET'
    pos = net.cell_types[cell_name].pos
    net.cell_types[cell_name] = pyramidal_l5ET(
        cell_name=_short_name(cell_name), pos=pos)
    
    net.gid_ranges['L5_ET'] = range(170, 170+et_it_ratio)
    
    cell_name = 'L5_IT'
    pos = net.cell_types[cell_name].pos
    net.cell_types[cell_name] = pyramidal_l5IT(
        cell_name=_short_name(cell_name), pos=pos)
    
    net.gid_ranges['L5_IT'] = range(170+et_it_ratio+1, 270)


    # change positions
    shuffled_pos = random.sample(net.pos_dict['L5_pyramidal'], len(net.pos_dict['L5_pyramidal']))
    net.pos_dict['L5_pyramidal'].pos_dict = shuffled_pos

    cell_name = 'L5_basket'
    pos = net.cell_types[cell_name].pos
    net.cell_types[cell_name] = interneuron(
        cell_name=_short_name(cell_name), pos=pos)
    
    # L2/3
    cell_name = 'L2_pyramidal'
    pos = net.cell_types[cell_name].pos
    net.cell_types[cell_name] = pyramidal_l23(
        cell_name=_short_name(cell_name), pos=pos)
    
    cell_name = 'L2_basket'
    pos = net.cell_types[cell_name].pos
    net.cell_types[cell_name] = interneuron(
        cell_name=_short_name(cell_name), pos=pos)
    
    return net