from sensitivity_analysis.get_acq_settings import get_marvy_sampling_settings, get_acq_settings_vfa, get_sampling_settings_vfa, \
    get_marvy_acq_settings
from sensitivity_analysis.config import get_sensitivity_config
from sensitivity_analysis.sensitivity_functions import calc_df_error_of_optimized_acq_settings, calc_df_fit_error, \
    calc_error_all_acq_settings_on_grid
from sensitivity_analysis.plotting_functions import create_fit_error_plot_optimized, create_fit_error_plot_vfa, \
    create_fit_error_plot_marvy
from sensitivity_analysis.optimization_functions import get_min_loss_acq_settings


def calc_and_save_error_df_for_plotting():
    cfg = get_sensitivity_config()

    nr_steps = cfg["nr_steps"]
    nr_avg = cfg["nr_avg"]
    nr_steps_optimization_per_param = cfg["nr_steps_optimization_per_param"]
    SNR = cfg["SNR"]

    gt_relax_ranges = {
        "M0": cfg["gt_relax_ranges"]["M0"],
        "T1": [*cfg["gt_relax_ranges"]["T1"], nr_steps],
        "T2": [*cfg["gt_relax_ranges"]["T2"], nr_steps],
    }

    noise_properties = {
        "type": "gaussian",
        "SNR": SNR,
    }

    dict_fit_ranges = cfg["fit_ranges"]
    out = cfg["output_paths"]

    calc_df_fit_error(get_acq_settings_vfa(), get_sampling_settings_vfa(), gt_relax_ranges, noise_properties, nr_avg,
                          dict_fit_ranges, save_path=out["vfa"])
    calc_df_fit_error(get_marvy_acq_settings(), get_marvy_sampling_settings(), gt_relax_ranges, noise_properties, nr_avg,
                          dict_fit_ranges, save_path=out["marvy"])
    calc_error_all_acq_settings_on_grid(nr_steps_optimization_per_param)
    dict_best_acq_settings_T1, dict_best_acq_settings_T2 = get_min_loss_acq_settings()
    calc_df_error_of_optimized_acq_settings(dict_best_acq_settings_T1, get_marvy_sampling_settings(), gt_relax_ranges,
                                            noise_properties,
                                            nr_avg, dict_fit_ranges,
                                            out["optimized_T1"])
    calc_df_error_of_optimized_acq_settings(dict_best_acq_settings_T2, get_marvy_sampling_settings(), gt_relax_ranges,
                                            noise_properties,
                                            nr_avg, dict_fit_ranges,
                                            out["optimized_T2"])

def plot_errors_of_dfs():
    create_fit_error_plot_vfa()
    create_fit_error_plot_marvy()
    dict_best_acq_settings_T1, dict_best_acq_settings_T2 = get_min_loss_acq_settings()
    create_fit_error_plot_optimized('T1', dict_best_acq_settings_T1)
    create_fit_error_plot_optimized('T2', dict_best_acq_settings_T2)
    return


if __name__ == "__main__":
    calc_and_save_error_df_for_plotting()
    plot_errors_of_dfs()
