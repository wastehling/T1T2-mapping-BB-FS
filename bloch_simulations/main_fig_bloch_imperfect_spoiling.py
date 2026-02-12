import numpy as np
import matplotlib.pyplot as plt
from bloch_simulations.create_figures import plot_signal_evolution_over_time_different_spin_ensembles, plot_signal_over_delta_phi


def create_imperfect_spoiling_figure():
    acq_settings = {
        'sim_steps': 1000,
        'TR': 5,
        'flip_angle':  25,
        'T1_discrete': [500, ],
        'T2_discrete': [20],
        'k': 30,  # TFE factor
        'dur_spir': 100,
        'dur_spoil': 10,
        'dur_bb': 12,
    }

    spoil_settings = {
        'perfect_spoiling': False,
        'nr_spins': 360,
        'delta_phi_discrete': [117, 150],
        # 'sim_range_delta_phi': np.arange(0, 180, 0.2) #Used for paper
        'sim_range_delta_phi': np.arange(120, 180, 0.5) #For fast easy visualization
    }

    additional_settings = {
        'SIM_TFE': True,
        'SIM_BB': True,
    }

    plot_settings={}
    plot_settings['colors'] = [
        (38,70,83),
        (42,157,143),
        (233,196,106),
        (244,162,97),
        (231,111,81),
        (140,146,172),
        (82,113,157),
        (23,55,94)
    ]
    #based on https://coolors.co/palette/264653-2a9d8f-e9c46a-f4a261-e76f51
    for i in range(len(plot_settings['colors'])):
        r, g, b = plot_settings['colors'][i]
        plot_settings['colors'][i] = (r / 255., g / 255., b / 255.)

    #create empty figure with one column and two lines to give to functions to fill
    fig, axes = plt.subplots(2,2, figsize=(18, 10), sharex=False)


    plot_signal_evolution_over_time_different_spin_ensembles({**acq_settings, 'flip_angle': 5, },
                                                             spoil_settings,
                                                             {**plot_settings, 'xlim': [4310, 4780], 'ylim': [0.025, 0.068]},
                                                             axes[0,0], **additional_settings)
    plot_signal_evolution_over_time_different_spin_ensembles({**acq_settings, 'flip_angle': 15,
                                                              # 'T1_discrete': [1500, ], 'T2_discrete': [50]
                                                              },
                                                             spoil_settings,
                                                             {**plot_settings, 'xlim': [4060, 4510],
                                                              # 'ylim': [0.02, 0.037]
                                                              'ylim': [0.0548, 0.0596]
                                                              },
                                                             axes[0,1], **additional_settings)


    plot_signal_over_delta_phi({**acq_settings, 'flip_angle': 5, },
                               spoil_settings, plot_settings, axes[1,0], **additional_settings)

    plot_signal_over_delta_phi({**acq_settings, 'flip_angle': 15,
                                    # 'T1_discrete': [1500, ], 'T2_discrete': [50]
                                },
                               spoil_settings, plot_settings, axes[1,1], **additional_settings)


    #set manual spacing
    plt.subplots_adjust(hspace=0.4, left=0.075, right=0.98, top=0.936, bottom=0.1)
    #add a), b) c), d) to the subplots
    x_off = -0.15
    axes[0, 0].text(x_off, 1.05, 'a)', transform=axes[0, 0].transAxes, fontsize=16, fontweight='bold')
    axes[0, 1].text(x_off, 1.05, 'b)', transform=axes[0, 1].transAxes, fontsize=16, fontweight='bold')
    axes[1, 0].text(x_off, 1.05, 'c)', transform=axes[1, 0].transAxes, fontsize=16, fontweight='bold')
    axes[1, 1].text(x_off, 1.05, 'd)', transform=axes[1, 1].transAxes, fontsize=16, fontweight='bold')

    path_save = './figures/figS1_signal_evolution_bloch_spoiling.svg'
    plt.savefig(path_save, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    create_imperfect_spoiling_figure()
