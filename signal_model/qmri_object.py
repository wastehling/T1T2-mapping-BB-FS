import torch
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import curve_fit
from signal_model.vfa import vfa_parameters_to_signal
from signal_model.bb_vfa import bb_vfa_parameters_to_signal

class qmri_object:
    def __init__(self, param):
        self.model = param['model']
        self.parameter_ranges = param['parameter_ranges']
        if self.model == 'VFA':
            self.TR = param['TR']


        elif self.model == 'BB_VFA':
            self.TR = param['TR']
            self.k = param['k']
            self.n = (param['n']-1) % self.k +1


        else:
            raise ValueError("Invalid parameter. Choose eiter 'IR' or 'VFA'.")




    def get_model_type(self):
        return self.model

    def get_signal(self, *args):
        # 3 args for IR, VFA and interleaved_VFA: self, xval, M0, T1
        # 5 args for BB_VFA: self, xval, T2prep, M0, T1, T2


        if len(args) == 3:
            M0 = args[1]
            T1 = args[2]
            FA = args[0]
            _, sig = vfa_parameters_to_signal(torch.tensor(M0), torch.tensor(T1), torch.tensor(self.TR),
                                              torch.tensor(FA))
            return sig

        elif len(args) == 4:
            if self.model == 'BB_VFA':
                FA = torch.tensor(args[0][0])
                dur_spir = torch.tensor(args[0][1])
                dur_bb = torch.tensor(args[0][2])
                dur_spoil = torch.tensor(args[0][3])
                M0 = torch.tensor(args[1])
                T1 = torch.tensor(args[2])
                T2 = torch.tensor(args[3])
                sig = bb_vfa_parameters_to_signal(
                    M0,
                    T1,
                    T2,
                    torch.tensor(self.TR).clone().detach(),
                    torch.tensor(dur_spir).clone().detach(),
                    dur_bb,
                    torch.tensor(dur_spoil).clone().detach(),
                    torch.tensor(FA).clone().detach(),
                    torch.tensor(self.n).clone().detach(),
                    torch.tensor(self.k).clone().detach()
                )
                return sig
            else:
                raise ValueError("5 input parameters but model is not BB_VFA.")
        else:
            raise ValueError("Invalid number of parameters.")

    def get_parameter_ranges(self):
        return self.parameter_ranges


    def fit_least_squares_array(self, xvalues, signal, njobs):
        try:
            if self.model in {'VFA'}:
                def parfun(i):
                    return self.fit_least_squares(xvalues[i, :], signal[i, :])
            elif self.model == 'BB_VFA':
                def parfun(i):
                    return self.fit_least_squares([xvalues[0][i, :], xvalues[1][i, :], xvalues[2][i, :],
                                                   xvalues[3][i, :]], signal[i, :])

            output = Parallel(n_jobs=njobs)(
                delayed(parfun)(i) for i in tqdm(range(len(signal)), position=0, leave=True))
            if self.model in {'VFA'}:
                M0, T1, pcov = np.transpose(output)
            elif self.model == 'BB_VFA':
                M0, T1, T2, stddev_M0, stddev_T1, stddev_T2 = np.transpose(output)


        except:
            M0 = np.zeros(len(signal))
            T1 = np.zeros(len(signal))
            if self.model == 'BB_VFA':
                T2 = np.zeros(len(signal))
                stddev_M0 = np.zeros(len(signal))
                stddev_T1 = np.zeros(len(signal))
                stddev_T2 = np.zeros(len(signal))

            for i in range(len(signal)):
                if self.model in {'VFA'}:
                    M0[i], T1[i] = self.fit_least_squares(xvalues[i,:], signal[i, :])
                elif self.model == 'BB_VFA':
                    M0[i], T1[i], T2[i], stddev_M0[i], stddev_T1[i], stddev_T2[i] = self.fit_least_squares( [xvalues[0][i,:], xvalues[1][i,:], xvalues[2][i,:], xvalues[3][i,:]], signal[i, :])
        if self.model in {'VFA'}:
            return [M0, T1]
        elif self.model == 'BB_VFA':
            return [M0, T1, T2, stddev_M0, stddev_T1, stddev_T2]



    def fit_least_squares(self, xvalues, dw_data):
        bounds = self.parameter_ranges
        # bounds are rescaled such that each parameter changes at roughly the same rate to help fitting.
        if self.model in {'VFA'}:
            bounds = ([bounds['M0'][0], bounds['T1'][0]],
                  [bounds['M0'][1], bounds['T1'][1]])
            # ini_guess_M0 = bounds[0][0] + (bounds[1][0] - bounds[0][0]) / 2
            idx_min_flip_angle = np.argmin(xvalues)
            ini_guess_M0 = dw_data[idx_min_flip_angle] / np.sin(np.deg2rad(xvalues[idx_min_flip_angle]))
            if ini_guess_M0 < bounds[0][0] or ini_guess_M0 > bounds[1][0]:
                ini_guess_M0 = bounds[0][0] + (bounds[1][0] - bounds[0][0]) / 2

            ini_guess_T1 = bounds[0][1] + (bounds[1][1] - bounds[0][1]) / 2
            ini_guess = [ini_guess_M0, ini_guess_T1]
        elif self.model == 'BB_VFA':
            bounds = ([bounds['M0'][0], bounds['T1'][0], bounds['T2'][0]],
                      [bounds['M0'][1], bounds['T1'][1], bounds['T2'][1]])
            #ini_guess_M0: signal at lowest FA divided by sin of that flipangle
            idx_min_flip_angle = np.argmin(xvalues[0])
            ini_guess_M0 = dw_data[idx_min_flip_angle] / np.sin(np.deg2rad(xvalues[0][idx_min_flip_angle]))
            if ini_guess_M0 < bounds[0][0] or ini_guess_M0 > bounds[1][0]:
                ini_guess_M0 = bounds[0][0] + (bounds[1][0] - bounds[0][0]) / 2


            ini_guess_T1 = bounds[0][1] + (bounds[1][1] - bounds[0][1]) / 2
            ini_guess_T2 = bounds[0][2] + (bounds[1][2] - bounds[0][2]) / 2
            ini_guess = [ini_guess_M0, ini_guess_T1, ini_guess_T2]

        params, pcov = curve_fit(self.get_signal, xvalues, dw_data, p0=ini_guess, bounds=bounds, maxfev=100000)

        # correct for the rescaling of parameters
        if self.model in {'VFA'}:
            M0, T1 = params[0], params[1]
            return M0, T1
        elif self.model == 'BB_VFA':
            M0, T1, T2 = params[0], params[1], params[2]
            stddev_M0, stddev_T1, stddev_T2 = np.sqrt(np.diag(pcov))
            return M0, T1, T2, stddev_M0, stddev_T1, stddev_T2
        else:
            return 0

# you can run this script as main to plot the signal evolution for the different models
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fa = 15
    M0 = 1
    T1 = 500
    T2 = 50
    dur_spir = 10
    dur_bb = 12
    dur_spoil = 0
    TFE = 5
    TR = 6.6
    signal_fat_vfa = 0
    signal_bb_vfa = 0
    signal_bb_vfa_switched = 0
    for n in range(1, 3*TFE):
        qmri_fat = qmri_object(
            {'model': 'interleaved_VFA', 'parameter_ranges': {}, 'TR': TR, 'TR2': dur_spir,
             'n': np.mod(n,TFE), 'k': TFE}
                    )
        qmri_bb = qmri_object(
            {'model': 'BB_VFA', 'parameter_ranges': {},
             'TR': TR,   'n': np.mod(n, TFE), 'k': TFE}
                    )
        qmri_bb_switched = qmri_object(
            {'model': 'BB_VFA', 'parameter_ranges': {},
             'TR': TR,   'n': np.mod(n, TFE), 'k': TFE}
        )

        signal_bb_vfa = np.append(signal_bb_vfa, qmri_bb.get_signal([fa, dur_spir, dur_bb, dur_spoil], M0, T1,T2).numpy())
        signal_bb_vfa_switched = np.append(signal_bb_vfa_switched, qmri_bb_switched.get_signal([fa, dur_spoil, dur_bb,
                                                                                                dur_spir], M0, T1,T2).numpy())

    signal_bb_vfa = signal_bb_vfa[1:]
    signal_bb_vfa_switched = signal_bb_vfa_switched[1:]
    plt.plot(signal_bb_vfa, label='bb')
    plt.plot(signal_bb_vfa_switched, label='bb_switched')

    plt.legend()
    plt.xlabel('n')
    plt.ylabel('signal')
    plt.ylim(0, 0.15)
    plt.show()
