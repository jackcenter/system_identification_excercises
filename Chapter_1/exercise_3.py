import matplotlib.pyplot as plt
import numpy as np

from ResistorExperiment import ResistorExperiment
from ResistorData import ResistorData


def main():
     print_header()
     run_exercise_3()
     return 0


def print_header():
    print("=====================================================")
    print("                     Exercise 3:")
    print("Impact of noise on the regressor (input) measurements")
    print("=====================================================")
    print()


def run_exercise_3():
    print("Running Experiment 3...")
    np.random.seed(3)

    number_of_simulations = 100000
    iterations = [100]
    current_max = 0.01           # Amps
    resistance_actual = 1000     # Ohms

    experiments_00a = []
    experiments_00b = []
    experiments_05 = []
    experiments_10 = []

    for n in iterations:
        experiments_00a.append(Experiment_3(number_of_simulations, n, current_max, resistance_actual, 0.0))
        experiments_00a[-1].run_experiment()

        experiments_00b.append(Experiment_3(number_of_simulations, n, current_max, resistance_actual, 0.0))
        experiments_00b[-1].run_experiment()

        experiments_05.append(Experiment_3(number_of_simulations, n, current_max, resistance_actual, 0.0005))
        experiments_05[-1].run_experiment()

        experiments_10.append(Experiment_3(number_of_simulations, n, current_max, resistance_actual, 0.001))
        experiments_10[-1].run_experiment()

    print("Results:")
    print()
    print("         \tMean\tStandard Deviation")
    print("(0, 0)_a:\t{}\t{}".format(round(experiments_00a[-1].get_mean()), round(experiments_00a[-1].get_standard_deviation())))
    print("(0, 0)_b:\t{}\t{}".format(round(experiments_00b[-1].get_mean()), round(experiments_00b[-1].get_standard_deviation())))
    print("(0, 0.5):\t{}\t{}".format(round(experiments_05[-1].get_mean()), round(experiments_05[-1].get_standard_deviation())))
    print("(0, 1):  \t{}\t{}".format(round(experiments_10[-1].get_mean()), round(experiments_10[-1].get_standard_deviation())))
    print()

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.suptitle("Figure 1-4: pdf of the estimates")
    fig.supxlabel("R")

    axs[0].hist(experiments_00a[-1].get_regression_values(), bins=100, density=True, histtype='step')
    axs[0].hist(experiments_00b[-1].get_regression_values(), bins=100, density=True, histtype='step')

    axs[1].hist(experiments_00a[-1].get_regression_values(), bins=100, density=True, histtype='step')
    axs[1].hist(experiments_05[-1].get_regression_values(), bins=100, density=True, histtype='step')

    axs[2].hist(experiments_00a[-1].get_regression_values(), bins=100, density=True, histtype='step')
    axs[2].hist(experiments_10[-1].get_regression_values(), bins=100, density=True, histtype='step')

    print("Close plot to continue.")
    print()
    
    plt.show()


class Experiment_3 ( ResistorExperiment ):

    def __init__(self, simulations, iterations, current_max, resistance_actual, i_std):
        super().__init__(simulations, iterations, current_max, resistance_actual)
        
        self.i_std = i_std

    def generate_current_samples(self):
        return np.random.uniform(-self.i_max, self.i_max, self.iters)
        # noise = np.random.normal(0, self.i_std, self.iters)

    def generate_noise_samples(self, std_dev):
        return np.random.normal(0, std_dev, self.iters)
    
    def generate_voltage_samples(self, i_samples, n_samples):
        return np.array([self.r_act * i + n for i, n in zip(i_samples, n_samples)])

    def run_simulation(self):

        i_samples = self.generate_current_samples()
        n_samples = self.generate_noise_samples(1)
        v_samples = self.generate_voltage_samples(i_samples, n_samples)

        n_samples = self.generate_noise_samples(self.i_std)
        i_samples += n_samples

        r_value = self.calculate_least_squares_estimate(i_samples, v_samples)

        return ResistorData(i_samples, n_samples, v_samples, r_value)        


if __name__ == "__main__":
    main()
