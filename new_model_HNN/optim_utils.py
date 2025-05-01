
import hnn_core


def set_params(net, params):

    # Proximal 1
    synaptic_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                     'L5_basket': 1., 'L5_pyramidal': 1.}
    
    weights_ampa_p1 = {'L2_basket':
                       params['evprox1_ampa_L2_basket'],
                       'L2_pyramidal':
                       params['evprox1_ampa_L2_pyramidal'],
                       'L5_basket':
                       params['evprox1_ampa_L5_basket'],
                       'L5_pyramidal':
                       params['evprox1_ampa_L5_pyramidal']}
    weights_nmda_p1 = {'L2_basket':
                       params['evprox1_nmda_L2_basket'],
                       'L2_pyramidal':
                       params['evprox1_nmda_L2_pyramidal'],
                       'L5_basket':
                       params['evprox1_nmda_L5_basket'],
                       'L5_pyramidal':
                       params['evprox1_nmda_L5_pyramidal']}

    
    net.add_evoked_drive('evprox1',
                         mu=16,
                         sigma=2.3,
                         numspikes=1,
                         location='proximal',
                         weights_ampa=weights_ampa_p1,
                         weights_nmda=weights_nmda_p1,
                         synaptic_delays=synaptic_delays_p, event_seed=544)

    # Distal
    synaptic_delays_d = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 1.}
    
    weights_ampa_d = {'L2_basket':
                       params['evdist_ampa_L2_basket'],
                       'L2_pyramidal':
                       params['evdist_ampa_L2_pyramidal'],
                       'L5_pyramidal':
                       params['evdist_ampa_L5_pyramidal']}
    weights_nmda_d = {'L2_basket':
                       params['evdist_nmda_L2_basket'],
                       'L2_pyramidal':
                       params['evdist_nmda_L2_pyramidal'],
                       'L5_pyramidal':
                       params['evdist_nmda_L5_pyramidal']}
    

    net.add_evoked_drive('evdist',
                         mu=58.5,
                        sigma=2.5,
                        numspikes=1,
                         location='distal',
                         weights_ampa=weights_ampa_d,
                         weights_nmda=weights_nmda_d,
                         synaptic_delays=synaptic_delays_d, event_seed=544)


    weights_ampa_p2 = {'L2_basket':
                       params['evprox2_ampa_L2_basket'],
                       'L2_pyramidal':
                       params['evprox2_ampa_L2_pyramidal'],
                       'L5_basket':
                       params['evprox2_ampa_L5_basket'],
                       'L5_pyramidal':
                       params['evprox2_ampa_L5_pyramidal']}
    weights_nmda_p2 = {'L2_basket':
                       params['evprox2_nmda_L2_basket'],
                       'L2_pyramidal':
                       params['evprox2_nmda_L2_pyramidal'],
                       'L5_basket':
                       params['evprox2_nmda_L5_basket'],
                       'L5_pyramidal':
                       params['evprox2_nmda_L5_pyramidal']}
    
    net.add_evoked_drive('evprox2',
                        mu=97,
                        sigma=10,
                        numspikes=6,
                        location='proximal',
                        weights_ampa=weights_ampa_p2,
                        weights_nmda=weights_nmda_p2,
                        synaptic_delays=synaptic_delays_p, event_seed=544)
    
    delay = net.delay