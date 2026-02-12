import numpy as np
import matplotlib.pyplot as plt
import tqdm
from bloch_simulations.bloch_sim import get_ffe_signal_perfect_spoiling, calc_magn_evolution, get_signal_evol_bb, \
    get_idx_of_first_pulse_from_last_simulated_shot
from signal_model.qmri_object import qmri_object

fontsize = 16
alpha = 0.8
markersize = 15

def update_min_max_signal(signal_evol, signal_min, signal_max):
    '''
    Update min and max signal value
    :param signal_evol: array of signal evolution
    :param signal_min: current min signal value
    :param signal_max: current max signal value
    :return: updated min and max signal value
    '''
    signal_evol = np.array(signal_evol)
    if signal_evol.min() < signal_min:
        signal_min = signal_evol.min()
    if signal_evol.max() > signal_max:
        signal_max = signal_evol.max()
    return signal_min, signal_max


def plot_signal_evolution_over_time_different_spin_ensembles(acq_settings, spoil_settings, plot_settings, axes, **kwargs, ):
    '''
    Plot signal evolution of different spin ensembles (T1_discrete, T2_discrete) over time
    :param acq_settings:
    :param spoil_settings:
    :param plot_settings:
    :param axes:
    :param kwargs:
    :return:
    '''
    colors = plot_settings.get('colors')
    idx_color = 2
    signal_min=1
    signal_max=0
    idx_steps_to_drop = 400  # drop first 200 samples to avoid transient state
    #add a plot_diff argument to kwargs to plot the difference between the simulated signal and the perfect spoiling signal
    kwargs['plot_diff'] = False

    for SIM_TFE in [True, False]:
        #create kwargs copy
        kwargs_discrete = kwargs.copy()
        kwargs_discrete['SIM_TFE'] = SIM_TFE
        # for T1 in acq_settings['T1_discrete']:
            # for T2 in acq_settings['T2_discrete']:
        T1 = acq_settings['T1_discrete'][0]  # only one T1 for now
        T2 = acq_settings['T2_discrete'][0]  # only one T2 for now
        for d_phi in spoil_settings['delta_phi_discrete']:
            acq_settings_discrete = acq_settings.copy()
            acq_settings_discrete['T1'] = T1
            acq_settings_discrete['T2'] = T2
            acq_settings_discrete.pop('T1_discrete', None)
            acq_settings_discrete.pop('T2_discrete', None)
            spoil_settings_discrete = spoil_settings.copy()
            spoil_settings_discrete['delta_phi'] = d_phi
            spoil_settings_discrete.pop('delta_phi_discrete', None)

            #Plot Bloch simulation
            signal_evol, arr_time_signal_sampled = calc_magn_evolution(acq_settings_discrete, spoil_settings_discrete, **kwargs_discrete)
            #drop first 200 samples to avoid transient state
            arr_time_signal_sampled = arr_time_signal_sampled[idx_steps_to_drop:]
            signal_evol = signal_evol[idx_steps_to_drop:]
            print(f'After dropping the first 200 samples due to transient state the sampled time starts at {arr_time_signal_sampled[0]}ms and ends at {arr_time_signal_sampled[-1]}ms')
            if kwargs_discrete.get('SIM_TFE') is True:
                # lbl = f'T1: {T1}ms, T2: {T2}ms, delta_phi: {spoil_settings_discrete["delta_phi"]}deg, TFE'
                lbl = f'I-SPGR, $\\Delta_\\phi$: {spoil_settings_discrete["delta_phi"]}$^\circ$'
                marker = 'x'
                ref_signal, time_ref_signal_sampled = get_signal_evol_bb(acq_settings_discrete, spoil_settings_discrete, **kwargs_discrete)
                ref_signal = ref_signal[idx_steps_to_drop:]
                time_ref_signal_sampled = time_ref_signal_sampled[idx_steps_to_drop:]
            else:
                # lbl = f'T1: {T1}ms, T2: {T2}ms, delta_phi: {spoil_settings_discrete["delta_phi"]}deg, FFE'
                lbl = f'SPGR, $\\Delta_\\phi$: {spoil_settings_discrete["delta_phi"]}$^\circ$'
                marker = 'o'
                ref_signal = get_ffe_signal_perfect_spoiling(acq_settings_discrete['TR'], acq_settings_discrete['T1'],
                                                              acq_settings_discrete['flip_angle'])
                time_ref_signal_sampled = np.arange(0, acq_settings_discrete['sim_steps'] * acq_settings_discrete['TR'], acq_settings_discrete['TR'])[200:]
                ref_signal = np.full_like(time_ref_signal_sampled, ref_signal, dtype=float)
            if kwargs_discrete.get('plot_diff') is True:
                # axes.scatter(time_ref_signal_sampled, ref_signal, label=lbl, color=colors[idx_color],
                #              marker=marker, s=markersize, alpha=alpha)
                axes.scatter(arr_time_signal_sampled, (signal_evol-ref_signal)/ref_signal,
                             label=f'Rel. diff to ref {lbl}', color=colors[idx_color],
                             marker=marker, s=markersize,
                             # alpha=alpha
                             )
                print(f'Max diff betweeen calculated timings is {np.max(np.array(time_ref_signal_sampled) - np.array(arr_time_signal_sampled))}, should be zero!')
            else:
                axes.scatter(arr_time_signal_sampled, signal_evol, label=lbl, color=colors[idx_color],
                      marker=marker, s=markersize, alpha=alpha)

            idx_color += 1
            if idx_color % len(colors) == 0:
                idx_color = 0

        # simulate TFE with bloch simulation
        if kwargs_discrete.get('SIM_TFE') is True:
            lbl = f'I-SPGR (perfect spoiling)'
            marker = 'x'
            signal_evol, arr_time_signal_sampled = calc_magn_evolution(acq_settings_discrete,
                                                                       {**spoil_settings_discrete, 'perfect_spoiling': True},
                                                                       **kwargs_discrete)

            axes.scatter(arr_time_signal_sampled, signal_evol, label=lbl, color='black',
                         marker=marker, s=markersize, alpha=alpha)

        signal_min, signal_max = update_min_max_signal(signal_evol, signal_min, signal_max)


        # #plot evolution of I-SPGR signal equation
        # if kwargs_discrete.get('plot_diff') is False:
        #     if kwargs_discrete.get('SIM_TFE') is True:
        #         signal_ISPGR, timing_signal_sampled_bb_ss = get_signal_evol_bb(acq_settings_discrete, spoil_settings_discrete, **kwargs_discrete)
        #         axes.scatter(timing_signal_sampled_bb_ss, signal_ISPGR, label=f'I-SPGR-Signal-Equation',
        #                  # color=colors[idx_color],
        #                      color='black',
        #                      marker='x', s=markersize)
        #         signal_min, signal_max = update_min_max_signal(signal_ISPGR, signal_min, signal_max)

    if kwargs_discrete.get('plot_diff') is False:
        signal_SPGR = get_ffe_signal_perfect_spoiling(acq_settings_discrete['TR'], acq_settings_discrete['T1'],
                                                          acq_settings_discrete['flip_angle'])
        axes.axhline(y=signal_SPGR,  label=f'SPGR (perfect spoiling)', color=colors[idx_color])
        signal_min, signal_max = update_min_max_signal(signal_SPGR, signal_min, signal_max)


    last_sim_timestep = arr_time_signal_sampled[-1]
    axes.set_xlim([last_sim_timestep-500, last_sim_timestep])

    # axes.set_xlim([2000, 2500])
    #check if xlime and ylim are given in plot_settings, if so use them
    if 'xlim' in plot_settings:
        axes.set_xlim(plot_settings['xlim'])
    if 'ylim' in plot_settings:
        axes.set_ylim(plot_settings['ylim'])
    # axes.set_ylim([signal_min - 0.1 * (signal_max - signal_min), signal_max + 0.1 * (signal_max - signal_min)])
    # axes.set_ylim([signal_min*0.9, signal_max*1.1])
    #fontsize ticks
    axes.tick_params(axis='both', which='major', labelsize=fontsize)




    axes.set_xlabel('Time [ms]', fontsize=fontsize)
    axes.set_ylabel('Signal [a.u.]', fontsize=fontsize)
    axes.set_title(f'Signal evolution of spin ensamble. '
                   # f'(T1: {T1}ms, T2: {T2}ms) \n'
                   f'Flip angle: {acq_settings_discrete["flip_angle"]}$^\circ$',
                   # f'TR: {acq_settings_discrete["TR"]}ms, Flip angle: {acq_settings_discrete["flip_angle"]}$^\circ$,'
                   # f'$T_{{\\mathrm{{Fat-saturation}}}}$: {acq_settings_discrete["dur_spir"]}',
                   fontsize=fontsize)

    handles, labels = axes.get_legend_handles_labels()
    order = [0,4,5,3,1,2]
    axes.legend([handles[idx] for idx in order], [labels[idx] for idx in order],)


def plot_signal_over_delta_phi(acq_settings, spoil_settings, plot_settings, axes, **kwargs):
    colors = plot_settings.get('colors')
    idx_color = 0
    markersize = 5
    for SIM_TFE in [False, True]:
        #create kwargs copy
        kwargs_discrete = kwargs.copy()
        kwargs_discrete['SIM_TFE'] = SIM_TFE

        for T1 in acq_settings['T1_discrete']:
            for T2 in acq_settings['T2_discrete']:
                acq_settings_discrete = acq_settings.copy()
                acq_settings_discrete['T1'] = T1
                acq_settings_discrete['T2'] = T2
                acq_settings_discrete.pop('T1_discrete', None)
                acq_settings_discrete.pop('T2_discrete', None)

                last_avg_signal = []
                if kwargs_discrete.get('SIM_TFE') is True:
                    first_signal_from_last_shot = []

                for delta_phi in tqdm.tqdm(spoil_settings['sim_range_delta_phi']):
                    spoil_settings_discrete = spoil_settings.copy()
                    spoil_settings_discrete['delta_phi'] = delta_phi
                    signal_evol, arr_time_signal_sampled = calc_magn_evolution(acq_settings_discrete,
                                                                               spoil_settings_discrete, **kwargs_discrete)
                    if kwargs_discrete.get('SIM_TFE') is True:
                        idx_last_simulated_first_pulse = get_idx_of_first_pulse_from_last_simulated_shot(acq_settings_discrete.get('sim_steps'), acq_settings_discrete.get("k"))
                        first_signal_from_last_shot.append(signal_evol[idx_last_simulated_first_pulse])
                        lbl = f'I-SPGR signal (Bloch simulation, first signal from shot)'
                        last_avg_signal.append(signal_evol[idx_last_simulated_first_pulse])
                        marker = 'v'

                    else:
                        idx_last_simulated_first_pulse = -1
                        lbl = f'SPGR signal (Bloch simulation)'
                        last_avg_signal.append(signal_evol[idx_last_simulated_first_pulse])
                        marker = '^'

                #simulation done

                axes.plot(spoil_settings['sim_range_delta_phi'], last_avg_signal, color=colors[idx_color], label=lbl,
                          marker=marker, markersize=markersize-2, linestyle='dashed', alpha=alpha)
                if kwargs_discrete.get('SIM_TFE') is True:
                    # axes.plot(spoil_settings['sim_range_delta_phi'], first_signal_from_last_shot, color=colors[idx_color],
                    #           marker='x', markersize=4, linestyle='dashed',
                    #           label=f'Signal Spin Ensamble (first Pulse after last gap) (T1 = {acq_settings_discrete["T1"]}ms, T2 = {acq_settings_discrete["T2"]}ms)')

                    qmri_bb = qmri_object(
                        {'model': 'BB_VFA', 'parameter_ranges': {},
                         'TR': float(acq_settings_discrete['TR']), 'n': 1,
                         'k': acq_settings_discrete.get('k')}
                    )
                    if kwargs_discrete.get('SIM_BB') is True:
                        dur_bb = acq_settings_discrete.get('dur_bb')
                        dur_spoil = acq_settings_discrete.get('dur_spoil')
                    else:
                        dur_bb = 0
                        dur_spoil = 0
                    M0 = 1
                    transient_state_signal_first_pulse = qmri_bb.get_signal(
                        [acq_settings_discrete['flip_angle'], acq_settings_discrete.get('dur_spir'), dur_bb, dur_spoil], M0,
                        acq_settings_discrete['T1'], acq_settings_discrete['T2']).numpy()
                    axes.axhline(y=transient_state_signal_first_pulse, color=colors[idx_color],  linestyle='dashed',
                                 label=f'I-SPGR (perfect spoiling, first signal from shot)')
                else: #sim TFE wrong
                    signal_SPGR = get_ffe_signal_perfect_spoiling(acq_settings['TR'], acq_settings_discrete['T1'],
                                                                  acq_settings['flip_angle'])
                    axes.axhline(y=signal_SPGR, color=colors[idx_color], linestyle='dashed',
                                 label=f'SPGR signal (perfect spoiling)')

                idx_color += 1
                if idx_color % len(colors) == 0:
                    idx_color = 0



    axes.set_title(f'Signal of spin ensamble. '
                   # f' (T1: {acq_settings_discrete["T1"]}ms, T2: {acq_settings_discrete["T2"]}ms) \n '
                   f'Flip angle: {round(acq_settings["flip_angle"],1)}$^\circ$', fontsize=fontsize)
    # f'TR: {acq_settings["TR"]}ms, Flip angle: {round(acq_settings["flip_angle"], 1)}$^\circ$', fontsize = fontsize)
    axes.set_xlabel('Phase increment $\\Delta_\\phi$ [$^\circ$]', fontsize=fontsize)
    axes.set_ylabel('Signal [a.u.]', fontsize=fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize)

    axes.set_xlim([0, 180])

    handles, labels = axes.get_legend_handles_labels()
    order = [1, 0, 3, 2]
    axes.legend([handles[idx] for idx in order], [labels[idx] for idx in order], )
    # axes.legend()