from signal_model.qmri_object import qmri_object
import numpy as np
import matplotlib.pyplot as plt

LOW_BOUND_T2 = 0
HIGH_BOUND_T2 = 1e3
LOW_BOUND_T1 = 0
HIGH_BOUND_T1 = 5e3
LOW_BOUND_M0 = 0
HIGH_BOUND_M0 = 1e6


def get_qmri_object(model_name, dict_acq_settings, dict_fitting_settings=None):
    T1_range = [LOW_BOUND_T1, HIGH_BOUND_T1]
    T2_range = [LOW_BOUND_T2, HIGH_BOUND_T2]
    M0_range = [LOW_BOUND_M0, HIGH_BOUND_M0]

    if dict_fitting_settings and 'T2_range' in dict_fitting_settings:
        T2_range = dict_fitting_settings.get('T2_range')
    if dict_fitting_settings and 'M0_range' in dict_fitting_settings:
        M0_range = dict_fitting_settings.get('M0_range')
    if dict_fitting_settings and 'T1_range' in dict_fitting_settings:
        T1_range = dict_fitting_settings.get('T1_range')

    if model_name == 'BB_VFA':
        qmri = qmri_object({'model': 'BB_VFA',
                            'TR': dict_acq_settings.get('TR'),
                            'parameter_ranges': {'M0': M0_range,
                                                 'T1': T1_range, 'T2': T2_range},
                            'n': dict_acq_settings.get('n'),
                            'k': dict_acq_settings.get('k')
                            })
        return qmri

    elif model_name == 'VFA':
        qmri = qmri_object({'model': 'VFA',
                            'TR': dict_acq_settings.get('TR'),
                            'parameter_ranges': {
                                'M0': M0_range,
                                'T1': T1_range,
                            },
                            })
        return qmri
    else:
        raise ValueError('Model name not recognized')


def get_signal_for_given_fa(model_name, sampled_fa, dict_relax_prop_gt, dict_acq_settings):
    if model_name == 'BB_VFA':
        qmri = get_qmri_object(model_name, dict_acq_settings)
        signal = qmri.get_signal(
            [sampled_fa,
             dict_acq_settings.get('dur_spir'),
             dict_acq_settings.get('dur_bb'),
             dict_acq_settings.get('dur_spoil')],
            dict_relax_prop_gt.get('M0'), dict_relax_prop_gt.get('T1'), dict_relax_prop_gt.get('T2'))
        return signal
    elif model_name == 'VFA':
        qmri = get_qmri_object(model_name, dict_acq_settings)
        signal = qmri.get_signal(sampled_fa, dict_relax_prop_gt.get('M0'), dict_relax_prop_gt.get('T1'))
        return signal
    else:
        raise ValueError('Model name not recognized')


def get_signals_and_xvals_sampled_for_differnt_acq_dur(model_name, dict_acq_settings, dict_relax_prop_gt, dict_samplings):
        signals_sampled = []
        for i in range(len(dict_samplings.get('flip_angle'))):
            copy_dict_acq_settings = dict_acq_settings.copy()
            copy_dict_acq_settings['dur_bb'] = dict_samplings.get('dur_bb')[i]
            signals_sampled.append(get_signal_for_given_fa(model_name, dict_samplings.get('flip_angle')[i], dict_relax_prop_gt, copy_dict_acq_settings))

        signals_sampled = np.stack([np.array(signals_sampled)])
        xvals = [
            np.stack([dict_samplings.get('flip_angle')]),
            np.stack([[dict_acq_settings.get('dur_spir')]*len(dict_samplings.get('flip_angle'))]),
            np.stack([dict_samplings.get('dur_bb')]),
            np.stack([[dict_acq_settings.get('dur_spoil')]*len(dict_samplings.get('flip_angle'))])
        ]
        return xvals, signals_sampled


def get_signal_for_all_fas(model_name, dict_relax_prop_gt, dict_acq_settings):
    signals = []
    for sampled_fa in range(0, 90):
        signals.append(get_signal_for_given_fa(model_name, sampled_fa, dict_relax_prop_gt, dict_acq_settings))
    return signals



def get_normalized_signal_over_all_fa(dict_acq_settings, T1, T2):
    qmri = get_qmri_object('BB_VFA', dict_acq_settings)
    signal = qmri.get_signal([np.linspace(0, 90, 91), dict_acq_settings.get('dur_spir'), dict_acq_settings.get('dur_bb'), dict_acq_settings.get('dur_spoil')], 1, T1, T2)
    signal = np.array(signal)
    signal_normalized = signal / np.max(signal)
    return signal_normalized


def get_normalized_signal_at_fa(dict_acq_settings, T1, T2, fa):
    signal_all_fa = get_normalized_signal_over_all_fa(dict_acq_settings, T1, T2)
    signal_at_fa = signal_all_fa[fa]
    return signal_at_fa


def add_gaussian_noise(signals_sampled, noise_properties):
    SNR = noise_properties.get('SNR')
    # sigma = signals_sampled/SNR
    sigma = np.max(signals_sampled) / SNR

    noise = np.random.normal(0, sigma, signals_sampled.shape)
    signals_sampled_noisy = signals_sampled + noise

    return signals_sampled_noisy


def add_rician_noise(signals_sampled, noise_properties):
    SNR = noise_properties.get('SNR')
    sigma = signals_sampled/SNR

    noise1 = np.random.normal(0, sigma, signals_sampled.shape)
    noise2 = np.random.normal(0, sigma, signals_sampled.shape)
    signals_sampled_noisy = np.sqrt((signals_sampled + noise1) ** 2 + noise2 ** 2)

    return signals_sampled_noisy


if __name__ == '__main__':
    #short visualization
    acq_set = {'TR': 3.5, 'dur_spir': 200, 'dur_spoil': 10,
                         'dur_bb': 12, 'n': 1, 'k': 30}
    dict_relax_prop_gt = {'M0': 1e5, 'T1': 1000, 'T2': 100}

    fig = plt.figure()
    signal_spge = np.array(get_signal_for_all_fas('VFA', dict_relax_prop_gt, acq_set))
    plt.plot(signal_spge, label='SPGE')
    signal_ispge = np.array(get_signal_for_all_fas('BB_VFA', dict_relax_prop_gt, acq_set))
    plt.plot(signal_ispge, label='I-SPGE')

    plt.title(f"Acq. settings: TR: {acq_set['TR']}ms, Dur-Spir: {acq_set['dur_spir']}ms, Dur-BB: {acq_set['dur_bb']}ms\n"
              f" Dur-Spoil: {acq_set['dur_spoil']}ms, n: {acq_set['n']}, k: {acq_set['k']} ")
    plt.legend()
    plt.xlabel('FA')
    plt.ylabel('Signal')
    plt.show()
