import matplotlib.pyplot as plt
from bloch_simulations.bloch_sim import get_ffe_signal_perfect_spoiling, calc_magn_evolution, get_signal_evol_bb


def plot_signal_evolution_over_time_different_spin_ensembles(acq_settings, spoil_settings, plot_settings, **kwargs, ):
    '''
    Plot signal evolution of different spin ensembles (T1_discrete, T2_discrete) over time
    :param acq_settings:
    :param spoil_settings:
    :param plot_settings:
    :param axes:
    :param kwargs:
    :return:
    '''
    # fig, axes = plt.subplots(figsize=(15, 6))
    fig, axes = plt.subplots(figsize=(8.27, 4)) #8.27 is word horizontal A4 size in inches
    marker_size_bloch_simulation = 12
    marker_size_signalequation = 12
    colors = plot_settings.get('colors')
    idx_color = 0
    alpha = 1
    fontsze = 11
    fontsze_legend = 6
    symbl_empty, = axes.plot([0], marker='None', linestyle='None', label='dummy-tophead')

    lbls_txt = []
    lbls_symb = []
    for T1 in acq_settings['T1_discrete']:
        for idx_T2, T2 in enumerate(acq_settings['T2_discrete']):
            for d_phi in spoil_settings['delta_phi_discrete']:
                acq_settings_discrete = acq_settings.copy()
                acq_settings_discrete['T1'] = T1
                acq_settings_discrete['T2'] = T2
                acq_settings_discrete.pop('T1_discrete', None)
                acq_settings_discrete.pop('T2_discrete', None)
                spoil_settings_discrete = spoil_settings.copy()
                spoil_settings_discrete['delta_phi'] = d_phi
                spoil_settings_discrete.pop('delta_phi_discrete', None)

                lbls_txt.append(f'T1: {T1}ms, T2: {T2}ms')
                lbls_symb.append(symbl_empty)

                #Plot bloch simulation
                signal_evol, arr_time_signal_sampled = calc_magn_evolution(acq_settings_discrete, spoil_settings_discrete, **kwargs)
                # take only every second point
                signal_evol = signal_evol[::2]
                arr_time_signal_sampled = arr_time_signal_sampled[::2]
                # p1, = axes.plot(arr_time_signal_sampled, signal_evol, label=f'Bloch simulation', color=colors[idx_color],
                #                 marker='o', markersize=3, linestyle='dashed', alpha=alpha)
                p1 = axes.scatter(arr_time_signal_sampled, signal_evol, label=f'Bloch simulation',
                                color=colors[idx_color],
                               facecolor='none', edgecolor=colors[idx_color],
                                marker='o',  alpha=alpha, s=marker_size_bloch_simulation,
                                   )

                lbls_symb.append(p1)
                lbls_txt.append('Bloch simulation')

                #plot evolution of I-SPGR signal equation
                if kwargs.get('SIM_TFE') is True:
                    signal_evol_bb_ss, timing_signal_sampled_bb_ss = get_signal_evol_bb(acq_settings_discrete, spoil_settings_discrete, **kwargs)
                    #take only every second point
                    signal_evol_bb_ss = signal_evol_bb_ss[::2]
                    timing_signal_sampled_bb_ss = timing_signal_sampled_bb_ss[::2]
                    p1 = axes.scatter(timing_signal_sampled_bb_ss, signal_evol_bb_ss, label=f'I-SPGR-Signal',
                                    color=colors[idx_color], alpha=alpha, marker='|', s=marker_size_signalequation,)
                    lbls_symb.append(p1)
                    lbls_txt.append('I-SPGR-Signal')

                #plot steady state signal
                if idx_T2 == 0:
                    steady_state_signal = get_ffe_signal_perfect_spoiling(acq_settings_discrete['TR'], acq_settings_discrete['T1'],
                                                                      acq_settings_discrete['flip_angle'])
                    p1, = axes.plot(arr_time_signal_sampled, [steady_state_signal]*len(arr_time_signal_sampled), label=f'SPGR-Signal', color=colors[idx_color],
                                    linestyle='--', alpha=alpha, marker='None', markersize=marker_size_signalequation)
                    lbls_txt.append('SPGR-Signal')
                    lbls_symb.append(p1)
                else:
                    #append a dummy symbol for the same T1 but different T2
                    lbls_txt.append('')
                    lbls_symb.append(symbl_empty)

                idx_color += 1
                if idx_color % len(colors) == 0:
                    idx_color = 0

    #xlim and ylim
    axes.set_xlim([0, arr_time_signal_sampled[-1]])
    axes.set_ylim([0, 0.105])
    #ticks fontsize
    axes.tick_params(axis='both', which='major', labelsize=fontsze)

    axes.set_xlabel('Time [ms]', fontsize=fontsze)
    axes.set_ylabel('Signal [a.u.]', fontsize=fontsze)
    axes.set_title(
    f"Signal evolution (Perfect spoiling) \n"
    f"TR: {acq_settings['TR']}ms, Flip angle: ${acq_settings['flip_angle']}^\\circ$, "
    f"k: {acq_settings.get('k')}, "
    f"$T_{{FS}}$: {acq_settings.get('dur_spir', 0)}ms, "
    f"$T_{{BS}}$: {acq_settings.get('dur_bb', 0)}ms, "
    f"$T_{{Sp}}$: {acq_settings.get('dur_spoil', 0)}ms",
    fontsize = fontsze
    )
    axes.legend(lbls_symb, lbls_txt, loc='upper right', fontsize=fontsze_legend, frameon=False, ncol=4)
    #set borders left and right and wspace
    plt.subplots_adjust(left=0.09, right=0.98, top=0.885, bottom=0.15, wspace=0.2)
    plt.savefig('./figures/fig3_signal_evol_bloch.svg', dpi=300)
    plt.show()


def get_acq_spoil_and_plot_settings():
    acq_settings = {
        'sim_steps': 130,
        'TR': 5,
        'flip_angle': 15,
        'T1_discrete': [500, 1500],
        'T2_discrete': [25, 50],
        'k': 20,
        'n': 1,
        'dur_spir': 50,
        'dur_spoil': 6,
        'dur_bb': 12,
    }

    spoil_settings = {
        'perfect_spoiling': True,
        'delta_phi_discrete': [150],
    }

    additinal_settings = {
        'SIM_TFE': True,
        'SIM_BB': True,
    }

    plot_settings = {}
    plot_settings['colors'] = [
        (38, 70, 83),
        (42, 157, 143),
        (233, 196, 106),
        (244, 162, 97),
        (231, 111, 81),
        (140, 146, 172),
        (82, 113, 157),
        (23, 55, 94)]
    # based on https://coolors.co/palette/264653-2a9d8f-e9c46a-f4a261-e76f51

    # Normalize the RGB values to [0, 1]
    for i in range(len(plot_settings['colors'])):
        r, g, b = plot_settings['colors'][i]
        plot_settings['colors'][i] = (r / 255., g / 255., b / 255.)

    return acq_settings, spoil_settings, plot_settings, additinal_settings

def create_figure_perfect_spoiling_for_different_spin_ensamples():
    acq_settings, spoil_settings, plot_settings, additinal_settings = get_acq_spoil_and_plot_settings()
    plot_signal_evolution_over_time_different_spin_ensembles(acq_settings, spoil_settings, plot_settings, **additinal_settings)

if __name__ == "__main__":
    create_figure_perfect_spoiling_for_different_spin_ensamples()
