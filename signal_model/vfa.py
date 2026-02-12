import torch


def vfa_coefficients_to_parameters(coefficients, parameter_ranges, simulation_method='sigmoid'):
    """
    converts vfa coefficients to parameters (M0, TRoverT1)
    Args:
        coefficients: input coefficients
        parameter_ranges: ranges of output parameters to which input coefficients should be mapped
        simulation_method: method of mapping

    Returns:
        M0, TRoverT1: vfa parameters
    """

    # retrieve parameter constraints
    M0min, M0max = parameter_ranges['M0'][0], parameter_ranges['M0'][1]
    T1min, T1max = parameter_ranges['T1'][0], parameter_ranges['T1'][1]

    # coefficient to parameter
    if simulation_method == 'sigmoid':
        M0 = M0min + torch.sigmoid(coefficients[:, 0].unsqueeze(1)) * (M0max - M0min)
        T1 = T1min + torch.sigmoid(coefficients[:, 1].unsqueeze(1)) * (T1max - T1min)
    elif simulation_method == 'linear':
        M0 = M0min + (coefficients[:, 0].unsqueeze(1) * (M0max - M0min))
        T1 = T1min + (coefficients[:, 1].unsqueeze(1) * (T1max - T1min))
    else:
        raise NotImplementedError(f'only linear and sigmoidal simulation method available, not {simulation_method}')

    return M0, T1


def vfa_parameters_to_signal(M0, T1, TR, FA):
    """
    simulate VFA signal (Mz), following:
    Mz(FA) = M0*sin(FA)*(1-e^(-TR/T1))/(1-cos(FA)e^(-TR/T1))

    Args:
        M0: tensor of magnetization at thermal equilibirum
        TRoverT1: tensor of repetition time over T1
        FA: tensor of flip angle

    Returns:
        simulated_data: simulated VFA signal

    """
    TRoverT1 = torch.div(TR, T1)
    relative_Mz = torch.sin(torch.deg2rad(FA)) * (1 - torch.exp(-TRoverT1)) / \
                  (1 - torch.cos(torch.deg2rad(FA)) * torch.exp(-TRoverT1))
    Mz = relative_Mz * M0
    return relative_Mz, Mz