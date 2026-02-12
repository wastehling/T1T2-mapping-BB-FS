from sensitivity_analysis.main_sensitivity_simultaneous_T1T2 import calc_and_save_error_df_for_plotting, plot_errors_of_dfs
from bloch_simulations.main_fig_bloch_perfect_spoiling import create_figure_perfect_spoiling_for_different_spin_ensamples
from bloch_simulations.main_fig_bloch_imperfect_spoiling import create_imperfect_spoiling_figure

def main():
    # Perfect spoiling plot
    create_figure_perfect_spoiling_for_different_spin_ensamples()
    #imperfect spoiling plot
    create_imperfect_spoiling_figure()

    #Sensitivity functions
    calc_and_save_error_df_for_plotting() #Calc df to be plotted afterwards
    plot_errors_of_dfs()

if __name__ == "__main__":
    main()