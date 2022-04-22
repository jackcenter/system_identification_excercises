import matplotlib.pyplot as plt
import numpy as np

from ResistorExperiment import ResistorExperiment


def main():
    print_header()
    run_exercise_2()
    return 0


def print_header():
    print("=====================================================")
    print("                     Exercise 2:")
    print(" Study of the asymptotic distribution of an estimate ")
    print("=====================================================")
    print()


def run_exercise_2():
    print("Running Experiment 2...")
    np.random.seed(2)

    number_of_simulations = 100000
    iterations = [1, 2, 4, 8]
    current_actual = 0.01           # Amps
    resistance_actual = 1000        # Ohms

    experiments_a = []

    for n in iterations:
        experiments_a.append(Experiment_2a(number_of_simulations, n, current_actual, resistance_actual))
        experiments_a[-1].run_experiment()

    experiments_b = []

    for n in iterations:
        experiments_b.append(Experiment_2b(number_of_simulations, n, current_actual, resistance_actual))
        experiments_b[-1].run_experiment()

    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
    fig.suptitle("Figure 1-3: pdf of the estimates")
    fig.supxlabel("R")

    for i in range(0, 4):
        axs[i].hist(experiments_a[i].get_regression_values(), bins=100, density=True, histtype='step')
        axs[i].hist(experiments_b[i].get_regression_values(), bins=100, density=True, histtype='step')
        axs[i].set_ylabel("N = " + str(experiments_a[i].iters))

    print("Close plot to continue.")
    print()
    
    plt.show()


class Experiment_2a ( ResistorExperiment ):

    def __init__(self, simulations, iterations, current_max, resistance_actual):
        super().__init__(simulations, iterations, current_max, resistance_actual)

    def generate_current_samples(self):
        return self.i_max * np.ones(self.iters)

    def generate_noise_samples(self):
        return np.random.normal(0, 0.2, self.iters)
    
    def generate_voltage_samples(self, i_samples, n_samples):
        return np.array([self.r_act * i + n for i, n in zip(i_samples, n_samples)])


class Experiment_2b ( ResistorExperiment ):
    def __init__(self, simulations, iterations, current_max, resistance_actual):
        super().__init__(simulations, iterations, current_max, resistance_actual)

    def generate_current_samples(self):
        return self.i_max * np.ones(self.iters)

    def generate_noise_samples(self):
        n_max = 3**(1/2) * 0.2
        return np.random.uniform(-n_max, n_max, self.iters)
    
    def generate_voltage_samples(self, i_samples, n_samples):
        return np.array([self.r_act * i + n for i, n in zip(i_samples, n_samples)])


if __name__ == "__main__":
    main()
