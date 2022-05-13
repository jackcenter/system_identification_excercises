import matplotlib.pyplot as plt
import numpy as np

from Chapter_1.ResistorExperiment import ResistorExperiment
from Chapter_1.WeightedResistorData import WeightedResistorData


def main():
     print_header()
     results_combined = run_exercise_5a()
     run_exercise_5b(results_combined)
     return 0


def print_header():
    print("=====================================================")
    print("                     Exercise 5:")
    print("           Weighted least square estimation")
    print("=====================================================")
    print()


def run_exercise_5a(number_of_simulations=100000, show_plot=True):
    print("Running Experiment 5a...")
    np.random.seed(4)

    iterations = [100]
    current_actual = 0.01           # Amps
    resistance_actual = 1000        # Ohms

    experiments_low_noise = []
    experiments_high_noise = []

    for n in iterations:
        # Run low noise experiments (good sensor)
        experiments_low_noise.append(Experiment(number_of_simulations, n, current_actual, resistance_actual, 1))
        experiments_low_noise[-1].run_experiment()

        # Run high noise experiments (bad sensor)
        experiments_high_noise.append(Experiment(number_of_simulations, n, current_actual, resistance_actual, 4))
        experiments_high_noise[-1].run_experiment()

        results_low_noise = experiments_low_noise[-1].get_results()
        results_high_noise = experiments_high_noise[-1].get_results()

        results_combined = []
        
        # Combine data
        for data_low_noise, data_high_noise in zip(results_low_noise, results_high_noise):
            results_combined.append(WeightedResistorData.combine_two_data_sets(data_low_noise, data_high_noise))

        # Run LSR
        for datum in results_combined:
            datum.run_least_squares_regression()

        least_squares_regression_values_combined = np.array([datum.get_regression_value() for datum in results_combined])

        # Run WLS
        for datum in results_combined:    
            datum.run_weighted_least_squares()

        weighted_least_squares_values_combined = np.array([datum.get_regression_value() for datum in results_combined])

        print("Results:")
        print()
        print("               \tMean\tStandard Deviation")
        print("Low Noise:     \t{}\t{}".format(round(experiments_low_noise[-1].get_mean(), 2), round(experiments_low_noise[-1].get_standard_deviation(), 4)))
        print("High Noise:    \t{}\t{}".format(round(experiments_high_noise[-1].get_mean(), 2), round(experiments_high_noise[-1].get_standard_deviation(), 4)))
        print("Combined (LSR):\t{}\t{}".format(round(np.mean(least_squares_regression_values_combined), 2), round(np.std(least_squares_regression_values_combined), 4)))
        print("Combined (WLS):\t{}\t{}".format(round(np.mean(weighted_least_squares_values_combined), 2), round(np.std(weighted_least_squares_values_combined), 4)))
        print()

        if show_plot:
            fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
            fig.suptitle("Figure 1-6: Estimated Resistance Values for\nGood and Bad Sensors")
            fig.supxlabel("R")

            axs.hist(experiments_low_noise[-1].get_regression_values(), bins=100, density=True, histtype='step')
            axs.hist(experiments_high_noise[-1].get_regression_values(), bins=100, density=True, histtype='step') 
            axs.hist(weighted_least_squares_values_combined, bins=100, density=True, histtype='step')    

            plt.legend(["Low Noise", "High Noise", "Combined"])
            
            print("Close plot to continue.")
            print()

            plt.show()

        return results_combined


def run_exercise_5b(results_combined):
    print("Running Experiment 5a...")    
    i_samples = results_combined[-1].get_current_samples()
    weights = results_combined[-1].get_weights()

    lsr_value = calculate_theoretical_standard_deviation_lsr(i_samples, weights)
    wls_value = calculate_theoretical_standard_deviation_wls(i_samples, weights)

    print("Theoretical Results (final current sequence from 5a):")
    print()
    print("Least Squares Regression (LSR):\t{}".format(lsr_value))
    print("Weighted Least Squares (WLS):  \t{}".format(wls_value))
    print()

    return 0


class Experiment ( ResistorExperiment ):

    def __init__(self, simulations, iterations, current_max, resistance_actual, v_std):
        super().__init__(simulations, iterations, current_max, resistance_actual)

        self.v_std = v_std


    def generate_current_samples(self):
        return np.random.uniform(-self.i_max, self.i_max, self.iters)


    def generate_noise_samples(self):
        return np.random.normal(0, self.v_std, self.iters)
    

    def generate_voltage_samples(self, i_samples, n_samples):
        return np.array([self.r_act * i + n for i, n in zip(i_samples, n_samples)])


    def calculate_weights(self):
        return self.v_std**2 * np.ones(self.iters)


    def run_simulation(self):

        i_samples = self.generate_current_samples()
        n_samples = self.generate_noise_samples()
        v_samples = self.generate_voltage_samples(i_samples, n_samples)
        r_value = self.calculate_least_squares_estimate(i_samples, v_samples)
        weights = self.calculate_weights()

        return WeightedResistorData(i_samples, v_samples, weights, r_value)


def calculate_theoretical_standard_deviation_lsr(i_samples, weights):
    return (np.sum(weights * i_samples * i_samples) / np.sum(i_samples * i_samples)**2)**(1/2)


def calculate_theoretical_standard_deviation_wls(i_samples, weights):
    return (1 / np.sum(i_samples * i_samples / weights))**(1/2)


if __name__ == "__main__":
    main()
