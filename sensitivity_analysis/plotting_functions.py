import os
import math
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from sensitivity_analysis.get_acq_settings import get_sampling_settings_vfa, get_acq_settings_vfa, \
    get_marvy_acq_settings
from sensitivity_analysis.signal_functions import get_signal_for_given_fa, get_normalized_signal_over_all_fa
from sensitivity_analysis.sensitivity_functions import turn_range_dict_to_list
from sensitivity_analysis.config import get_sensitivity_config


def plot_signal_over_fa(dict_acq_settings, dict_relax_prop_gt):

    cont_signals_bbvfa = []
    cont_signals_vfa = []

    for sampled_fa in range(0, 90):
        cont_signals_bbvfa.append(get_signal_for_given_fa('BB_VFA', sampled_fa, dict_relax_prop_gt, dict_acq_settings))
        cont_signals_vfa.append(get_signal_for_given_fa('VFA', sampled_fa, dict_relax_prop_gt, dict_acq_settings))

    plt.plot(cont_signals_bbvfa, label='BBVFA')
    plt.plot(cont_signals_vfa, label='VFA')

    plt.legend()
    plt.xlabel('FA')
    plt.ylabel('Signal')
    plt.title('Signal evolution for different flip angles')
    plt.show()


def plot_gt_sampled_and_fitted_signals(signal_gt, xvals_sampled, signal_sampled, signal_fitted, title=None):
    plt.plot(signal_gt, color='red', label='Groundtruth Signal')
    plt.scatter(xvals_sampled, signal_sampled, color='black', label='Sampled signals', marker='x')
    plt.plot(signal_fitted, label='Fitted signal')
    plt.legend()
    plt.xlabel('FA')
    plt.ylabel('Signal')
    if title:
        plt.title(title)
    plt.show()


def plot_fitting_error_GTGTvsVAR(df_rel_error,
                                 dict_plotting = {
                                                'gt_on_xaxis': 'T1',
                                                'gt_on_yaxis': 'M0',
                                                'cbar_minmax': 0.5,
                                                'idx_snr_val': 1,
                                                'fitting_error_names': ['rel_error_M0', 'rel_error_T1', 'rel_error_ini_guess_M0']
                                 },
                                 title=None, block_fig=False, save_name=None, lblsize=14):
    
    to_plot_x_axis = dict_plotting.get('gt_on_xaxis')
    to_plot_y_axis = dict_plotting.get('gt_on_yaxis')
    cbar_minmax = dict_plotting.get('cbar_minmax')

    gt_range_y = np.sort(df_rel_error[to_plot_x_axis].unique())
    gt_range_x = np.sort(df_rel_error[to_plot_y_axis].unique())

    range_snr = df_rel_error['SNR'].unique()

    snr_selected = range_snr[dict_plotting.get('idx_snr_val')]

    rel_error = np.zeros((len(gt_range_y), len(gt_range_x)))

    names = dict_plotting.get('fitting_error_names')

    fig, axes = plt.subplots(1, len(names), figsize=(5*len(names), 6), sharex=True, sharey=True, constrained_layout=True)

    for idx_typ, ax in enumerate(axes):
        #fill data
        for i, gt_val_y in enumerate(gt_range_y):
            for j, gt_val_x in enumerate(gt_range_x):
                subset = df_rel_error[(df_rel_error[to_plot_x_axis] == gt_val_y) &
                                      (df_rel_error[to_plot_y_axis] == gt_val_x) &
                                      (df_rel_error['SNR'] == snr_selected)]
                # rel_error[i, j] = subset[names[idx_typ]].values[0]
                rel_error[i, j] = np.mean(subset[names[idx_typ]].values)

        dx = (gt_range_x[1] - gt_range_x[0]) / 2 if len(gt_range_x) > 1 else 0
        dy = (gt_range_y[1] - gt_range_y[0]) / 2 if len(gt_range_y) > 1 else 0
        extent = [gt_range_x.min() - dx, gt_range_x.max() + dx, gt_range_y.min() - dy, gt_range_y.max() + dy]
        img = axes[idx_typ].imshow(rel_error, aspect='auto', origin='lower',
                                   vmin=0, vmax= cbar_minmax, cmap='viridis',
                                    extent=extent
                                   )
        #print mean and std of rel_error
        mean_rel_error = np.mean(rel_error[rel_error > 0])
        print(f'Mean {names[idx_typ]}: {mean_rel_error:.4f}')
        print('Title was', title)
        print()

        axes[idx_typ].set_xticks(gt_range_x)

        if to_plot_y_axis == 'M0':
            axes[idx_typ].set_xlabel('Simulated Groundtruth M0', fontsize=lblsize)
            import matplotlib.ticker as ticker
            formatter = ticker.EngFormatter(unit='', places=0)  # e.g., 10.0k, 1.2M
            axes[idx_typ].xaxis.set_major_formatter(formatter)
        else:
            axes[idx_typ].set_xlabel(f'Simulated Groundtruth {to_plot_y_axis}\u2009[ms]', fontsize=lblsize)
            axes[idx_typ].set_xticklabels(gt_range_x.astype(int))

        for tick in axes[idx_typ].get_xticklabels():
            tick.set_rotation(45)
        
        #yticks
        if idx_typ == 0:
            axes[idx_typ].set_yticks(gt_range_y)
            axes[idx_typ].set_yticklabels(gt_range_y.astype(int))
            axes[idx_typ].set_ylabel(f'Simulated Groundtruth {to_plot_x_axis}\u2009[ms]', fontsize=lblsize)
            for tick in axes[idx_typ].get_yticklabels():
                tick.set_rotation(45)
        else:
            # axes[idx_typ].set_yticklabels([])
            axes[idx_typ].tick_params(labelleft=False)
        
        #title
        if names[idx_typ] == 'rel_error_M0':
            subtitle = 'Relative fitting error M0'
        elif names[idx_typ] == 'rel_error_T1':
            subtitle = 'Relative fitting error T1'
        elif names[idx_typ] == 'rel_error_T2':
            subtitle = 'Relative fitting error T2'
        else:
            subtitle = names[idx_typ]
        axes[idx_typ].set_title(subtitle, fontsize=lblsize)

    cbar = fig.colorbar(img, ax=axes, location='right', fraction=0.025, pad=0.02)
    cbar.set_label('Relative fitting error', fontsize=lblsize)

    if title:
        fig.suptitle(title, fontsize=lblsize)

    if save_name:
        plt.savefig(save_name)
    plt.show(block=block_fig)


def plot_loss_matrix_for_fixed_fit_range(loss_matrices, name_matrices, prediction, dict_relax_prop_gt, dict_relax_fit_range,
                                         save_fig_name=None):
    '''
    :param dict_acq_settings: settings of sequence
    :param dict_relax_prop_gt: M0, T1 and T2 have to be defined, ground truth values
    :param dict_samplings: FA and dur_bb have to be defined, where signal is sampled
    :param dict_relax_fit_range: M0_, T1_ and T2_ range have to be defined (2d arr with start and stop val) and
    nr_M0_range_values, nr_T1_range_values and nr_T2_range_values (int)
    :param SNR: noise level
    :return:
    '''
    if len(loss_matrices) == 4:
        print('Warning: loss_matrices is 4d, only using first 3 dimensions')

    fig, axes = plt.subplots(1, len(name_matrices), figsize=(6*len(name_matrices), 8))
    for idx, ax in enumerate(axes):

        if np.all(loss_matrices[idx] == 0):
            cax = axes[idx].imshow(loss_matrices[idx], aspect='auto', origin='lower',
                                   cmap='gray', norm=None)  # No normalization for all zeros
        else:
            cax = axes[idx].imshow(loss_matrices[idx], aspect='auto', origin='lower',
                                   cmap='gray',
                                   norm=LogNorm(vmin=np.min(loss_matrices[idx][loss_matrices[idx] > 0]),
                                                vmax=np.max(loss_matrices[idx])))
        #add red x at gt coordinates
        gt_T1_index = np.argmin(np.abs(turn_range_dict_to_list(dict_relax_fit_range,'T1_range') - dict_relax_prop_gt.get('T1')))  # Find index of closest T1
        gt_M0_index = np.argmin(np.abs(turn_range_dict_to_list(dict_relax_fit_range,'M0_range') - dict_relax_prop_gt.get('M0')))  # Find index of closest M0
        axes[idx].plot(gt_T1_index, gt_M0_index, 'rx', markersize=10, label='Ground Truth')

        #add actual fit
        fit_t1_index = np.argmin(np.abs(turn_range_dict_to_list(dict_relax_fit_range,'T1_range') - prediction.get('T1')))  # Find index of closest T1
        fit_m0_index = np.argmin(np.abs(turn_range_dict_to_list(dict_relax_fit_range,'M0_range') - prediction.get('M0')))
        axes[idx].plot(fit_t1_index, fit_m0_index, 'bx', markersize=10, label='Fitted')
        axes[idx].legend()


        axes[idx].set_xticks(np.arange(dict_relax_fit_range.get('nr_T1_range_values')))
        axes[idx].set_xticklabels(turn_range_dict_to_list(dict_relax_fit_range, 'T1_range').astype(int))
        for tick in axes[idx].get_xticklabels():
            tick.set_rotation(45)

        axes[idx].set_xlabel('Fixed T1')
        axes[idx].set_title(f'Loss matrix {name_matrices[idx]}')
        fig.colorbar(cax, ax=axes[idx], label=f'Loss', )

    axes[0].set_yticks(np.arange(dict_relax_fit_range.get('nr_M0_range_values')))
    axes[0].set_yticklabels(
        [f'{int(val):.0e}' for val in turn_range_dict_to_list(dict_relax_fit_range, 'M0_range')])
    axes[0].set_ylabel('Fixed M0')

    fig.suptitle(f'Loss matrices for different fixed T1 and M0 values \n '
                 f'GT (red x):  T1: {dict_relax_prop_gt.get("T1")}, M0: {int(dict_relax_prop_gt.get("M0")):.0e}, '
                 f'T2: {dict_relax_prop_gt.get("T2")} \n'
                 f'Fitted (blue x): T1: {math.ceil(prediction.get("T1"))}, M0: {int(prediction.get("M0")):.0e}, '
                 f'T2: {math.ceil(prediction.get("T2"))}')
    if save_fig_name:
        plt.savefig(save_fig_name)
    plt.show()


    
def plot_normalized_signals_for_two_different_acq_settings(T1_values, T2_values, dict_acq_settings_marvy, dict_acq_settings_optimized, titles,
                                                           save_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for T1 in T1_values:
        for T2 in T2_values:
            signal_normalized_marvy = get_normalized_signal_over_all_fa(dict_acq_settings_marvy, T1, T2)
            signal_normalized_optimized = get_normalized_signal_over_all_fa(dict_acq_settings_optimized, T1, T2)
            
            axes[0].plot(np.linspace(0, 90, 91), signal_normalized_marvy, label=f'T1: {T1} ms, T2: {T2} ms')
            axes[1].plot(np.linspace(0, 90, 91), signal_normalized_optimized, label=f'T1: {T1} ms, T2: {T2} ms')
    
    xlim_max = 30
    axes[0].set_xlim([0, xlim_max])
    axes[1].set_xlim([0, xlim_max])
    axes[0].set_xlabel('Flip angle (degrees)')
    axes[1].set_xlabel('Flip angle (degrees)')
    axes[0].set_ylabel('Normalized signal')
    axes[0].set_title(f'{titles[0]} settings \n TR: {dict_acq_settings_marvy.get("TR")}s, '
                      f'dur_spir: {dict_acq_settings_marvy.get("dur_spir")}ms,  '
                      f'dur_spoil: {dict_acq_settings_marvy.get("dur_spoil")}ms, '
                      f'dur_bb: {dict_acq_settings_marvy.get("dur_bb")}ms')
    axes[1].set_title(f'{titles[1]} settings \n TR: {dict_acq_settings_optimized.get("TR")}s, '
                      f'dur_spir: {dict_acq_settings_optimized.get("dur_spir")}ms, '
                      f'dur_spoil: {dict_acq_settings_optimized.get("dur_spoil")}ms, '
                      f'dur_bb: {dict_acq_settings_optimized.get("dur_bb")}ms')

    axes[0].legend()
    axes[1].legend()
    if save_name:
        plt.savefig(save_name)
    plt.show()



def plot_normalized_signal_for_different_TR_and_dur_spir():
    TR_values = [3.5, 25]
    dur_spir_values = [20, 400]
    T1_values = [2000, 3000]

    fa_max = 30
    #plto in one plot
    fig, axes = plt.subplots(1, 1, figsize=(18, 6))
    for TR in TR_values:
        for dur_spir in dur_spir_values:
            for T1 in T1_values:
                signal_normalized = get_normalized_signal_over_all_fa({'TR': TR, 'dur_spir': dur_spir, 'dur_spoil': 0,
                                                                       'dur_bb': 12, 'n': 1, 'k': 30}, T1, 20)
                axes.plot(np.arange(fa_max), signal_normalized[0:fa_max], label=f'T1: {T1}ms, '
                                                                           f'TR: {TR}s, dur_spir: {dur_spir}ms')
    plt.legend()
    plt.xlabel('Flip angle (degrees)')
    plt.ylabel('Normalized signal')
    plt.title('Normalized signal for different TR and duration of spiral readout')
    plt.show()


def plot_fitting_error_VFA(df, dict_plot_looking, lblsize=14):
    df_avg = df.groupby(['T1', 'M0'], as_index=False)[['rel_error_T1', 'rel_error_M0']].mean()

    gt_T1 = df_avg['T1'].values
    gt_M0 = df_avg['M0'].values
    rel_err_T1 = df_avg['rel_error_T1'].values
    rel_err_M0 = df_avg['rel_error_M0'].values

    plt.figure(figsize=(10, 5))
    plt.suptitle(dict_plot_looking.get('title', 'Fitting error'), fontsize=lblsize)
    plt.subplot(1, 2, 2)
    plt.plot(gt_T1, rel_err_T1,)
    plt.xlabel('Simulated Ground Truth T1 [ms]', fontsize=lblsize)
    plt.ylabel('Relative Fitting Error T1', fontsize=lblsize)
    plt.grid()
    plt.ylim([0, 0.1])

    plt.subplot(1, 2, 1)
    plt.plot(gt_T1, rel_err_M0,)
    plt.xlabel('Simulated Ground Truth T1 [ms]', fontsize=lblsize)
    plt.ylabel('Relative Fitting Error M0', fontsize=lblsize)
    plt.ylim([0, 0.1])

    plt.grid()

    plt.subplots_adjust(bottom=0.1, top=0.885, left=0.08, right=0.99, wspace=0.3)
    if dict_plot_looking.get('save_name', None):
        plt.savefig(dict_plot_looking.get('save_name'))
    plt.show(block=dict_plot_looking.get('block_fig', False))


if __name__ == "__main__":
    cfg = get_sensitivity_config()
    df_rel_error = pd.read_pickle(cfg['output_paths']['marvy'])

    plot_fitting_error_GTGTvsVAR(df_rel_error, dict_plotting={
                                    'gt_on_xaxis': 'T1',
                                    'gt_on_yaxis': 'T2',
                                    'cbar_minmax': 0.5,
                                    'idx_snr_val': 0,
                                    'fitting_error_names': ['rel_error_M0', 'rel_error_T1', 'rel_error_T2', ],
                                    },
                                 title='testplot',
                                 block_fig=True)
    plot_fitting_error_GTGTvsVAR(df_rel_error, dict_plotting={
        'gt_on_xaxis': 'T1',
        'gt_on_yaxis': 'T2',
        'cbar_minmax': 0.5,
        'idx_snr_val': 0,
        'fitting_error_names': ['stddev_M0', 'stddev_T1', 'stddev_T2', ]
    })


def create_fit_error_plot_optimized(type_param, dict_acq_settings):
    cfg = get_sensitivity_config()
    df = pd.read_pickle(cfg['output_paths'][f'optimized_{type_param}'])

    plot_fitting_error_GTGTvsVAR(df, dict_plotting={
            'gt_on_xaxis': 'T1',
            'gt_on_yaxis': 'T2',
            'cbar_minmax': 0.6,
            'idx_snr_val': 0,
            'fitting_error_names': ['rel_error_M0', 'rel_error_T1', 'rel_error_T2', ],
        },
                                 title=f'Relative fitting error for I-SPGR, optimized for minimal {type_param} error:\n'
                                       f'TR: {int(dict_acq_settings["TR"])}ms, '
                                       f'$T_{{FS}}$: {int(dict_acq_settings["dur_spir"])}ms, '
                                       f'$T_{{Sp}}$: {dict_acq_settings["dur_spoil"]}ms, '
                                       f'$T_{{BS}}$: {dict_acq_settings["dur_bb"]}ms, '  
                                       f'sampled flip angles\u2009[deg]: {", ".join(str(item) for item in get_sampling_settings_vfa()["flip_angle"])} ',
                                 block_fig=True,
                                 save_name=os.path.join(f'{cfg["output_path_images"]}/figS4_rel_error_optimized_{type_param}.svg')
    )


def create_fit_error_plot_vfa():
    cfg = get_sensitivity_config()
    df = pd.read_pickle(cfg['output_paths']['vfa'])

    plot_fitting_error_VFA(df, dict_plot_looking={
                                    'title': 'Relative fitting error for SPGR\n'
                                                  f'TR: {get_acq_settings_vfa()["TR"]}ms, '
                                                  f'sampled flip angles\u2009[deg]: {", ".join(str(item) for item in get_sampling_settings_vfa()["flip_angle"])} ',
                                    'block_fig': True,
                                    'save_name': os.path.join(f'{cfg["output_path_images"]}/figs4_rel_error_vfa_t1m0.svg')
                                }
                           )


def create_fit_error_plot_marvy():
    cfg = get_sensitivity_config()
    df = pd.read_pickle(cfg['output_paths']['marvy'])
    plot_fitting_error_GTGTvsVAR(df, dict_plotting={
        'gt_on_xaxis': 'T2',
        'gt_on_yaxis': 'T1',
        'cbar_minmax': 0.6,
        'idx_snr_val': 0,
        'fitting_error_names': ['rel_error_M0', 'rel_error_T1', 'rel_error_T2', ],
    },
                                 title='Relative fitting error for I-SPGR:\n'
                                        f'TR: {get_marvy_acq_settings()["TR"]}ms, '
                                        f'$T_{{FS}}$: {get_marvy_acq_settings()["dur_spir"]}ms, '
                                        f'$T_{{Sp}}$: {get_marvy_acq_settings()["dur_spoil"]}ms, '
                                        f'$T_{{BS}}$: {get_marvy_acq_settings()["dur_bb"]}ms, '  
                                        # f'k: {get_marvy_acq_settings()["k"]}, n: {get_marvy_acq_settings()["n"]}, '
                                        f'sampled flip angles\u2009[deg]: {", ".join(str(item) for item in get_sampling_settings_vfa()["flip_angle"])} ',
                                 block_fig=True,
                                 save_name=os.path.join(f'{cfg["output_path_images"]}/figS2_rel_error_marvy_t1t2.svg'))
