import torch


def bb_vfa_parameters_to_signal(M0, T1, T2, TR, dur_spir, dur_bb, dur_spoil, FA, n, k):
    """
    BB VFA signal equation (Mz)

    Args:
        M0: tensor of magnetization at thermal equilibirum
        TR: tensor of repetition times
        T1: tensor of T1
        FA: tensor of flip angle
        n: tensor of amount of pulses after the interruption of the steady state the centre of k-space is measured
        k: tensor of lines per shot

    Returns:
        simulated_data: simulated VFA signal

    """
    FA_rad = torch.deg2rad(FA)
    E1 = torch.exp(-TR / T1)
    E1_dur_spir = torch.exp(-dur_spir / T1)
    E2_dur_bb = torch.exp(-dur_bb / T2)
    E1_dur_spoil = torch.exp(-dur_spoil / T1)

    first_summand = (1 - E1) * ( 1-(E1*torch.cos(FA_rad))**(n-1) ) / (1-E1*torch.cos(FA_rad))

    denominator = 1 - E1_dur_spoil * E2_dur_bb * E1_dur_spir   * ( E1*torch.cos(FA_rad) )**k
    nominator_1 = (1-E1_dur_spoil)
    nominator_2 = (1-E1_dur_spir) * E1_dur_spoil * E2_dur_bb
    bra = (1 - (E1*torch.cos(FA_rad))**k) / (1 - E1*torch.cos(FA_rad))
    nominator_3 = E1_dur_spoil * E2_dur_bb * E1_dur_spir * (1-E1) * bra
    second_summand = (nominator_1 + nominator_2 + nominator_3)/denominator * (E1*torch.cos(FA_rad))**(n-1)

    sig = torch.sin(FA_rad) * M0 * (first_summand + second_summand)
    return sig


