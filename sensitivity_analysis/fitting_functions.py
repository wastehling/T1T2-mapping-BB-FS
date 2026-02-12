from sensitivity_analysis.signal_functions import get_qmri_object, get_signal_for_all_fas, \
                                                   get_signals_and_xvals_sampled_for_differnt_acq_dur


def do_qmri_fiiting_and_get_prediction(model_name, dict_acq_settings, xvals, signals_sampled, dict_fit_ranges=None):
    """
    Takes model with acquisition settings and sampled signals and fits the model to the signals
    :param model_name:
    :param dict_acq_settings:
    :param xvals:
    :param signals_sampled:
    :param dict_fit_ranges:
    :return:
    """
    qmri_object = get_qmri_object(model_name, dict_acq_settings, dict_fit_ranges)
    prediction = {}
    prediction['M0'], prediction['T1'], prediction['T2'], \
    prediction['stddev_M0'], prediction['stddev_T1'], prediction['stddev_T2'] = \
    qmri_object.fit_least_squares_array(xvals, signals_sampled, 1)
    return prediction


def sample_signal_and_fit(dict_relax_settings, dict_acq_settings, dict_samplings, noise_properties, dict_fit_ranges=None):
    signal_gt = get_signal_for_all_fas('BB_VFA', dict_relax_settings, dict_acq_settings)
    xvals_sampled, signal_sampled = get_signals_and_xvals_sampled_for_differnt_acq_dur('BB_VFA', dict_acq_settings,
                                                                                       dict_relax_settings,
                                                                                       dict_samplings)
    prediction = do_qmri_fiiting_and_get_prediction('BB_VFA', dict_acq_settings, xvals_sampled, signal_sampled, dict_fit_ranges)
    signal_fitted = get_signal_for_all_fas('BB_VFA', prediction, dict_acq_settings)
    return signal_gt, xvals_sampled, signal_sampled, signal_fitted, prediction
