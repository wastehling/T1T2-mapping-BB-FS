import numpy as np
from scipy.optimize import minimize
import pandas as pd

from sensitivity_analysis.signal_functions import get_normalized_signal_at_fa, get_signals_and_xvals_sampled_for_differnt_acq_dur
from sensitivity_analysis.get_acq_settings import get_marvy_acq_settings, get_marvy_sampling_settings
from sensitivity_analysis.signal_functions import add_gaussian_noise
from sensitivity_analysis.fitting_functions import do_qmri_fiiting_and_get_prediction
from sensitivity_analysis.config import get_sensitivity_config

def calc_rel_error_for_prediction_and_gt(prediction, gt):
    """
    Calculate relative error for prediction and ground truth
    :param prediction: dict with keys 'M0', 'T1', 'T2'
    :param gt: dict with keys 'M0', 'T1', 'T2'
    :return: dict with keys 'M0', 'T1', 'T2'
    """
    rel_error = {}
    for key in ['M0', 'T1', 'T2']:
        rel_error[key] = (prediction[key] - gt[key]) / gt[key]
    return rel_error


def objective(params, T1_values, T2_values):
    '''
    Calculate distance of different signals with given acq settings (params) for different T1 and T2 values at fixed FA
    :param params: [TR, dur_spir, dur_spoil, dur_bb]
    :param T1_values: range of T1 values
    :param T2_values: range of T2 values
    :return: array of distance for each T1 and T2 value
    '''
    n = 1
    k = 30
    dict_acq_settings = {'TR': params[0], 'dur_spir': params[1], 'dur_spoil': params[2],
                         'dur_bb': params[3], 'n': n, 'k': k}
    print(dict_acq_settings)
    type = 'fit_w_noise' # 'euclidean' #

    if type == 'euclidean':
        fa_to_optimize = 5
        signals = np.zeros((len(T1_values), len(T2_values)))

        for i, T1 in enumerate(T1_values):
            for j, T2 in enumerate(T2_values):
                signals[i, j] = get_normalized_signal_at_fa(dict_acq_settings, T1, T2, fa_to_optimize)

        # Calculate the separability metric (e.g., Euclidean distance between curves)
        separability = 0
        for i in range(len(T1_values)):
            for j in range(len(T2_values)):
                for m in range(i + 1, len(T1_values)):
                    for n in range(j + 1, len(T2_values)):
                            separability += np.linalg.norm(signals[i, j] - signals[m, n])  # Euclidean distance
        print('Separability:', separability)
        return -separability  # We negate because we want to maximize separability

    elif type == 'fit_w_noise':
        M0 = 1e5
        _, loss_T1, _ = get_loss_for_acq_settings_and_gt(dict_acq_settings, T1_values, T2_values, M0)
        return loss_T1

    elif type == 'debug':
        return np.random.rand()

    else:
        raise ValueError('Type not recognized')


def find_optimum_acq_settings(debug=False, sim_noise=False):
    if debug:
        return get_marvy_acq_settings()

    T1_values = np.linspace(100, 3000, 4)  # Example T1 values
    T2_values = np.linspace(20, 50, 2)  # Example T2 values

    initial_params = [3.5, 200, 10, 12]  # TR, dur_spir, dur_spoil, dur_bb
    bounds = [(3, 50), (0, 400), (10, 50), (12, 30)]

    print('Starting optimization...')
    result = minimize(objective, initial_params, args=(T1_values, T2_values), method='L-BFGS-B', bounds=bounds)

    result.x = [round(x, 1) for x in result.x]
    dict_best_acq_settings = {'TR': result.x[0], 'dur_spir': result.x[1], 'dur_spoil': result.x[2],
                              'dur_bb': result.x[3]}
    print(f"Optimized acquisition settings: {dict_best_acq_settings}")

    dict_best_acq_settings['n'] = 1
    dict_best_acq_settings['k'] = 30

    return dict_best_acq_settings


def get_loss_for_acq_settings_and_gt(dict_acq_settings, T1_values, T2_values, M0, nr_avg=1):
    xvals_list = []
    sampled_signals_list = []
    gt = {'M0': np.array([]), 'T1': np.array([]), 'T2': np.array([])}

    for i, T1 in enumerate(T1_values):
        for j, T2 in enumerate(T2_values):
            for idx_avg in range(nr_avg):
                dict_relax_prop_gt = {'M0': M0, 'T1': T1, 'T2': T2}
                xvals, sampled_signals = get_signals_and_xvals_sampled_for_differnt_acq_dur(model_name='BB_VFA',
                                                                                            dict_acq_settings=dict_acq_settings,
                                                                                            dict_relax_prop_gt=dict_relax_prop_gt,
                                                                                            dict_samplings=get_marvy_sampling_settings())
                sampled_signals = add_gaussian_noise(sampled_signals, {'SNR': 30})
                xvals_list.append(xvals)
                sampled_signals_list.append(sampled_signals)
                # append M0, T1 and T2 values to gt dict
                gt['M0'] = np.append(gt.get('M0'), M0)
                gt['T1'] = np.append(gt.get('T1'), T1)
                gt['T2'] = np.append(gt.get('T2'), T2)

    xvals_arr = np.squeeze(np.array(np.transpose(xvals_list, (1, 0, 2, 3))))
    sampled_signals_arr = np.squeeze(np.array(sampled_signals_list))
    pred = do_qmri_fiiting_and_get_prediction('BB_VFA', dict_acq_settings, xvals_arr, sampled_signals_arr)

    rel_error = calc_rel_error_for_prediction_and_gt(pred, {'M0': gt.get('M0'), 'T1': gt.get('T1'), 'T2': gt.get('T2')})
    loss_M0, loss_T1, loss_T2 = np.mean(np.abs(rel_error.get('M0'))), np.mean(np.abs(rel_error.get('T1'))), np.mean(np.abs(rel_error.get('T2')))
    return loss_M0, loss_T1, loss_T2


def get_min_loss_acq_settings():
    #gridsearch_acq_settings() has to be run before so that pkl is available
    from sensitivity_analysis.config import get_sensitivity_config
    cfg = get_sensitivity_config()
    df = pd.read_pickle(cfg["output_paths"]["gridsearch_acq_set"])

    nr_steps = cfg["nr_steps"]
    idx_min_loss_T1 = df['loss_T1'].idxmin()
    idx_min_loss_T2 = df['loss_T2'].idxmin()
    idx_min_loss_M0 = df['loss_M0'].idxmin()
    dict_best_acq_settings_min_loss_T1 = {'TR': df['TR'][idx_min_loss_T1], 'dur_spir': df['dur_spir'][idx_min_loss_T1],
                              'dur_spoil': df['dur_spoil'][idx_min_loss_T1],
                              'dur_bb': df['dur_bb'][idx_min_loss_T1], 'n': df['n'][idx_min_loss_T1],
                              'k': df['k'][idx_min_loss_T1]}
    dict_best_acq_settings_min_loss_T2 = {'TR': df['TR'][idx_min_loss_T2], 'dur_spir': df['dur_spir'][idx_min_loss_T2],
                                'dur_spoil': df['dur_spoil'][idx_min_loss_T2],
                                'dur_bb': df['dur_bb'][idx_min_loss_T2], 'n': df['n'][idx_min_loss_T2],
                                'k': df['k'][idx_min_loss_T2]}
    return dict_best_acq_settings_min_loss_T1, dict_best_acq_settings_min_loss_T2


def calc_best_acq_settings_via_gridsearch():
    dict_best_acq_settings_T1, dict_best_acq_settings_T2 = get_min_loss_acq_settings()
    print('Best acquisition settings:', dict_best_acq_settings_T1)


if __name__ == "__main__":
    calc_best_acq_settings_via_gridsearch()
    best_acq_settings = find_optimum_acq_settings()
    print('Best acquisition settings:', best_acq_settings)
