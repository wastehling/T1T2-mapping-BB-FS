import numpy as np
import pandas as pd
from tqdm import tqdm
from sensitivity_analysis.signal_functions import add_rician_noise, get_signals_and_xvals_sampled_for_differnt_acq_dur, add_gaussian_noise
from sensitivity_analysis.fitting_functions import do_qmri_fiiting_and_get_prediction
from sensitivity_analysis.optimization_functions import get_loss_for_acq_settings_and_gt
from sensitivity_analysis.config import get_sensitivity_config


def turn_range_dict_to_list(dict_ranges, key):
    value = dict_ranges.get(key)
    nr_values = dict_ranges.get(f'nr_{key}_values')
    return np.linspace(value[0], value[1], nr_values)


def calc_fit_error_for_one_acq(M0, T1, T2, noise, nr_avg, dict_acq_settings, dict_samplings, model_name='BB_VFA',
                        dict_fit_ranges=None):
    """

    :param M0: gt M0 (one value)
    :param T1: gt T1 (one value)
    :param T2: gt T2 (one value)
    :param noise: dict with type and SNR
    :param nr_avg: int, number of averages to be simulated
    :param dict_acq_settings:
    :param dict_samplings:
    :param model_name:
    :param dict_fit_ranges:
    :return: df with gt values and relative fitting errors
    """
    dict_relax_prop_gt = {'M0': M0, 'T1': T1, 'T2': T2}
    xvals, signals_sampled = get_signals_and_xvals_sampled_for_differnt_acq_dur(
        model_name, dict_acq_settings, dict_relax_prop_gt, dict_samplings
    )

    mean_rel_error_T1 = []
    mean_rel_error_T2 = []
    mean_rel_error_M0 = []
    # initial_guess_M0 = []

    for _ in range(nr_avg):
        if noise.get('type') == 'rician':
            signals_sampled_with_noise = add_rician_noise(signals_sampled, noise)
        elif noise.get('type') == 'gaussian':
            signals_sampled_with_noise = add_gaussian_noise(signals_sampled, noise)
        elif noise.get('type') == 'no':
             signals_sampled_with_noise = signals_sampled
        else:
            raise ValueError('Noise type not recognized')
        prediction = do_qmri_fiiting_and_get_prediction(model_name, dict_acq_settings, xvals, signals_sampled_with_noise,
                                                        dict_fit_ranges=dict_fit_ranges)
        mean_rel_error_T1.append(np.abs((prediction.get('T1') - T1)) / T1)
        mean_rel_error_T2.append(np.abs((prediction.get('T2') - T2)) / T2)
        mean_rel_error_M0.append(np.abs((prediction.get('M0') - M0)) / M0)
        # initial_guess_M0.append((ini_guess_M0 - M0) / M0)

    bias_error_T1 = np.mean(mean_rel_error_T1)
    bias_error_T2 = np.mean(mean_rel_error_T2)
    bias_error_M0 = np.mean(mean_rel_error_M0)
    # bias_error_ini_guess_M0 = np.mean(initial_guess_M0)
    #calc std dev
    std_dev_error_T1 = np.std(mean_rel_error_T1)
    std_dev_error_T2 = np.std(mean_rel_error_T2)
    std_dev_error_M0 = np.std(mean_rel_error_M0)
    # std_dev_error_ini_guess_M0 = np.std(initial_guess_M0)

    return {
        'M0': M0,
        'T1': T1,
        'T2': T2,
        'SNR': noise.get('SNR'),
        'NoiseType': noise.get('type'),
        'rel_error_T1': bias_error_T1,
        'rel_error_T2': bias_error_T2,
        'rel_error_M0': bias_error_M0,
        # 'rel_error_ini_guess_M0': bias_error_ini_guess_M0,
        'stddev_T1': std_dev_error_T1,
        'stddev_T2': std_dev_error_T2,
        'stddev_M0': std_dev_error_M0,
        # 'stddev_ini_guess_M0': std_dev_error_ini_guess_M0
    }


def calc_df_of_fit_error_for_set_of_gt(dict_acq_settings, dict_samplings,
                           gt_relax_ranges,
                           dict_fit_ranges,
                           noise_properties = {'type': 'gaussian', 'SNR': 50},
                           nr_avg=10,
                           ):
    '''
    :param dict_acq_settings: such as TR, dur_spir, dur_spoil, dur_bb, n, k
    :param dict_samplings: such as FA, dur_bb
    :param gt_relax_ranges: dict with ranges for M0, T1, T2 (e.g., {'M0': [1e5, 1e5, 5], ...})
    :param dict_fit_ranges: dict with ranges for M0, T1, T2 (e.g., {'M0_range': [1e3, 1e7], ...})
    :param noise_properties: dict with type and SNR (e.g., {'type': 'gaussian', 'SNR': 50})
    :param nr_avg: int, number of averages to be simulated
    :return:
    '''
    # gt_M0 = np.linspace(gt_relax_ranges.get('M0')[0], gt_relax_ranges.get('M0')[1], gt_relax_ranges.get('M0')[2])
    gt_M0 = gt_relax_ranges.get('M0')
    gt_T1 = np.linspace(gt_relax_ranges.get('T1')[0], gt_relax_ranges.get('T1')[1], gt_relax_ranges.get('T1')[2])
    gt_T2 = np.linspace(gt_relax_ranges.get('T2')[0], gt_relax_ranges.get('T2')[1], gt_relax_ranges.get('T2')[2])

    rel_error = []
    for T1 in tqdm(gt_T1):
        for T2 in gt_T2:
            for M0 in gt_M0:
                rel_error.append(calc_fit_error_for_one_acq(M0, T1, T2, noise_properties, nr_avg,
                                 dict_acq_settings, dict_samplings,
                                 model_name='BB_VFA', dict_fit_ranges=dict_fit_ranges))
    rel_error = pd.DataFrame(rel_error)
    return rel_error


def calc_loss_matrices_for_fixed_fit_range(dict_relax_fit_range, dict_acq_settings, signals_sampled, xvals, model_name='BB_VFA'):
    m0_fit_range = turn_range_dict_to_list(dict_relax_fit_range, 'M0_range')
    t1_fit_range = turn_range_dict_to_list(dict_relax_fit_range, 'T1_range')
    t2_fit_range = turn_range_dict_to_list(dict_relax_fit_range, 'T2_range')

    loss_matrix_m0 = np.zeros((len(m0_fit_range), len(t1_fit_range), len(t2_fit_range)))
    loss_matrix_t1 = np.zeros((len(m0_fit_range), len(t1_fit_range), len(t2_fit_range)))
    loss_matrix_t2 = np.zeros((len(m0_fit_range), len(t1_fit_range), len(t2_fit_range)))
    for i, M0 in enumerate(m0_fit_range):
        for j, T1 in enumerate(t1_fit_range):
            for k, T2 in enumerate(t2_fit_range):

                dict_fit_range = {'M0_range': [m0_fit_range[i], m0_fit_range[i] + 1],
                                   'T1_range': [t1_fit_range[j], t1_fit_range[j] + 1],
                                   'T2_range': [t2_fit_range[k], t2_fit_range[k] + 1]}
                prediction = do_qmri_fiiting_and_get_prediction(model_name, dict_acq_settings, xvals, signals_sampled,
                                                            dict_fit_ranges=dict_fit_range)

                loss_matrix_m0[i, j, k] = prediction.get('stddev_M0')
                loss_matrix_t1[i, j, k] = prediction.get('stddev_T1')
                loss_matrix_t2[i, j, k] = prediction.get('stddev_T2')
    return loss_matrix_m0, loss_matrix_t1, loss_matrix_t2


def get_loss_matrices_and_prediction_for_fixed_fit_range(dict_acq_settings, dict_relax_prop_gt, dict_samplings, dict_relax_fit_range, SNR=50):
        model_name = 'BB_VFA'
        xvals, signals_sampled = get_signals_and_xvals_sampled_for_differnt_acq_dur(
            model_name, dict_acq_settings, dict_relax_prop_gt, dict_samplings
        )
        signals_sampled = add_gaussian_noise(signals_sampled, {'SNR': SNR})
        loss_matrix_m0, loss_matrix_t1, loss_matrix_t2 = calc_loss_matrices_for_fixed_fit_range(dict_relax_fit_range, dict_acq_settings,
                                                                            signals_sampled, xvals, model_name='BB_VFA')

        prediction = do_qmri_fiiting_and_get_prediction(model_name, dict_acq_settings, xvals, signals_sampled,
                                                        dict_fit_ranges=dict_relax_fit_range)

        return [loss_matrix_m0, loss_matrix_t1, loss_matrix_t2], ['stddev_M0', 'stddev_T1', 'stddev_T2'], prediction



def calc_error_all_acq_settings_on_grid(i):
    nr_items_per_variable = i

    M0 = 1e5
    T1_values = np.linspace(100, 3000, nr_items_per_variable)  # Example T1 values
    T2_values = np.linspace(10, 300, nr_items_per_variable)  # Example T2 values

    TR_values = np.linspace(3.5, 50, nr_items_per_variable)
    dur_spir_values = np.linspace(20, 400, nr_items_per_variable)
    n_values = [1]
    k = 30
    rel_error = []
    for TR in TR_values:
        for dur_spir in dur_spir_values:
            if k*TR + dur_spir < 300 or k*TR + dur_spir > 600:
                continue
            for n in n_values:
                dict_acq_settings = {'TR': TR, 'dur_spir': dur_spir, 'dur_spoil': 10,
                         'dur_bb': 12, 'n': n, 'k': k}

                loss_M0, loss_T1, loss_T2 = get_loss_for_acq_settings_and_gt(dict_acq_settings, T1_values, T2_values, M0,
                                                                             nr_avg=10)
                #fill into df
                rel_error.append(pd.DataFrame({'TR': [TR], 'dur_spir': [dur_spir], 'dur_spoil': [10],
                                   'dur_bb': [12], 'n': [n], 'k': [k], 'loss_M0': loss_M0, 'loss_T1': loss_T1,
                                   'loss_T2': loss_T2}))

    df = pd.concat(rel_error, ignore_index=True)
    cfg = get_sensitivity_config()
    df.to_pickle(cfg["output_paths"]["gridsearch_acq_set"])

    return df


def calc_df_error_of_optimized_acq_settings(dict_optimized_settings, dict_sampling, gt_relax_ranges, noise_properties,
                                            nr_avg, dict_fit_ranges, output_name):
    df_rel_error_optimized_grid = calc_df_of_fit_error_for_set_of_gt(dict_optimized_settings, dict_sampling,
                                                                        gt_relax_ranges=gt_relax_ranges,
                                                                        noise_properties=noise_properties,
                                                                        nr_avg=nr_avg, dict_fit_ranges=dict_fit_ranges)
    df_rel_error_optimized_grid.to_pickle(output_name)


def calc_df_fit_error(dict_acq_settings, dict_samplings, gt_relax_ranges, noise_properties, nr_avg, dict_fit_ranges,
                          save_path):
    df_rel_error_vfa = calc_df_of_fit_error_for_set_of_gt(
        dict_acq_settings,
        dict_samplings,
        gt_relax_ranges=gt_relax_ranges, noise_properties=noise_properties,
        nr_avg=nr_avg, dict_fit_ranges=dict_fit_ranges)
    df_rel_error_vfa.to_pickle(save_path)
