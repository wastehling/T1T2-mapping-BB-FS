import numpy as np
from bloch_simulations.matrix_operations import rotate_z, rotate_x, get_rotation_matrix, get_rel_matrix
from signal_model.qmri_object import qmri_object


def get_ffe_signal_perfect_spoiling(TR, T1, flip_angle):
    alpha_rad = np.deg2rad(flip_angle)
    return 1 * (1 - np.exp(-TR / T1)) / (1 - np.cos(alpha_rad) * np.exp(-TR / T1)) * np.sin(alpha_rad)


def calc_magn_evolution(acq_settings, spoil_settings, **kwargs):
    '''
    Bloch simulation of a spin ensamble with/without TFE and with/without spoiling
    :param acq_settings:
    :param spoil_settings:
    :param kwargs: 
    :return: 
    '''
    if spoil_settings.get('perfect_spoiling', True):
        magn_spin_ensamble = [[0, 0, 1]]
    else:
        magn_spin_ensamble = np.zeros((spoil_settings.get('nr_spins'), 3))
        magn_spin_ensamble[:, 2] = 1

        deph_angles_spin = np.linspace(0, 360, spoil_settings.get('nr_spins'))
        # create rotation matrices for each spin during TR due to  crusher gradients
        rotation_matrices_crusher = np.array([rotate_z(angle) for angle in deph_angles_spin])

    phi = 0
    signal_evol = []
    arr_time_signal_sampled = []
    time_sampling = 0

    mat_relax_TR = get_rel_matrix(acq_settings['T1'], acq_settings['T2'], acq_settings.get('TR'))
    if kwargs.get('SIM_TFE') == True:
        mat_relax_gap = get_rel_matrix(acq_settings['T1'], acq_settings['T2'], acq_settings.get('dur_spir'))
        if kwargs.get('SIM_BB') == True:
            mat_relax_bb = get_rel_matrix(acq_settings['T2'], acq_settings['T1'], acq_settings.get('dur_bb'))
            mat_relax_bbspoil = get_rel_matrix(acq_settings['T1'], acq_settings['T2'], acq_settings.get('dur_spoil'))

    for sim_step in np.arange(1, acq_settings.get('sim_steps') +1):
        # Apply RF pulse
        phi = phi + (sim_step-1) * spoil_settings.get('delta_phi')
        phi = phi % 360  # wrap around to keep phi in [0, 360)
        mat_RF_pulse = get_rotation_matrix(phi, acq_settings['flip_angle'])
        magn_spin_ensamble = magn_spin_ensamble @ mat_RF_pulse.T

        # Sample signal
        mean_all_spins = np.mean(magn_spin_ensamble, axis=0)
        signal_evol.append(np.linalg.norm(mean_all_spins[0:2]))
        arr_time_signal_sampled.append(time_sampling)

        #relaxation
        magn_spin_ensamble = magn_spin_ensamble @ mat_relax_TR.T
        magn_spin_ensamble += np.array([0, 0, 1]) * (1 - np.exp(-acq_settings.get('TR') / acq_settings['T1']))

        # spoiling
        if spoil_settings.get('perfect_spoiling', True):
            #set 0 and 1 elem of magn_spin_ensamble to 0
            magn_spin_ensamble[0][0:2] = [0, 0]
        else:
            #gradient spoiling
            magn_spin_ensamble = np.einsum('ijk,ik->ij', rotation_matrices_crusher, magn_spin_ensamble)

        time_sampling += acq_settings.get('TR')

        # check if we are in a TFE
        if kwargs.get('SIM_TFE') is True and sim_step % acq_settings.get('k') == 0 and sim_step != 0:
            magn_spin_ensamble = magn_spin_ensamble @ mat_relax_gap.T
            magn_spin_ensamble += np.array([0, 0, 1]) * (1 - np.exp(-acq_settings.get('dur_spir') / acq_settings['T1']))
            time_sampling += acq_settings.get('dur_spir')

            if kwargs.get('SIM_BB') == True:
                magn_spin_ensamble = magn_spin_ensamble @ mat_relax_bb.T
                magn_spin_ensamble = magn_spin_ensamble @ mat_relax_bbspoil.T
                magn_spin_ensamble += np.array([0, 0, 1]) * (
                        1 - np.exp(-acq_settings.get('dur_spoil') / acq_settings['T1']))
                # spoiling
                if spoil_settings.get('perfect_spoiling', True):
                    # set 0 and 1 elem of magn_spin_ensamble to 0
                    magn_spin_ensamble[0][0:2] = [0, 0]
                else:
                    magn_spin_ensamble = np.einsum('ijk,ik->ij', rotation_matrices_crusher, magn_spin_ensamble)
                time_sampling = time_sampling + acq_settings.get('dur_bb') + acq_settings.get('dur_spoil')

    return signal_evol, arr_time_signal_sampled


def get_signal_evol_bb(acq_settings, spin_settings, **kwargs):
    '''
    Get signal evolution of TFE sequence from signal equation and time of sampling
    :param acq_settings:
    :param spin_settings: 
    :param kwargs: 
    :return: 
    '''

    M0 = 1
    signal_bb_vfa = []
    arr_time_signal_sampled = []
    sampling_time = 0

    if kwargs.get('SIM_BB') is True:
        dur_bb = acq_settings.get('dur_bb')
        dur_spoil = acq_settings.get('dur_spoil')
    else:
        dur_bb = 0
        dur_spoil = 0

    for idx_RF_pulse in range(1, acq_settings.get('sim_steps')+1):
        qmri_bb = qmri_object(
            {'model': 'BB_VFA', 'parameter_ranges': {},
             'TR': float(acq_settings['TR']), 'n': idx_RF_pulse,
             'k': acq_settings.get('k')}
        )
        signal = qmri_bb.get_signal([acq_settings['flip_angle'], acq_settings.get('dur_spir'), dur_bb, dur_spoil], M0,
                                    acq_settings['T1'], acq_settings['T2']).numpy()
        signal_bb_vfa = np.append(signal_bb_vfa, signal)
        arr_time_signal_sampled.append(sampling_time)
        sampling_time += acq_settings['TR']
        if idx_RF_pulse % acq_settings.get('k') == 0 and idx_RF_pulse != 0:
            sampling_time += acq_settings.get('dur_spir')
            if kwargs.get('SIM_BB') == True:
                sampling_time += acq_settings.get('dur_bb') + acq_settings.get('dur_spoil')

    return signal_bb_vfa, arr_time_signal_sampled




def get_idx_of_first_pulse_from_last_simulated_shot(sim_steps, tfe_fac):
    nr_full_tfe_shots = int((sim_steps-1) / tfe_fac)
    if nr_full_tfe_shots*tfe_fac+1 <= sim_steps:
        #if the last simulated shot is a full TFE shot, return the last pulse of that shot
        return nr_full_tfe_shots*tfe_fac
    else:
        return (nr_full_tfe_shots-1)*tfe_fac
