import matplotlib.pyplot as plt
import numpy as np

from Chapter_1.ResistorExperiment import ResistorExperiment


def main():
    print_header()
    run_exercise_1a()
    run_exercise_1b()
    return 0


def print_header():
    print("=====================================================")
    print("                     Exercise 1:")
    print(" Least squares estimation of the value of a resistor ")
    print("=====================================================")
    print()


def run_exercise_1a(number_of_simulations=100, show_plot=True):
    print("Running Experiment 1a...")
    np.random.seed(26)

    iterations = [10, 100, 1000, 10000]
    current_max = 0.01              # Amps
    resistance_actual = 1000        # Ohms

    experiments = []
    for n in iterations:
        experiments.append(ResistorExperiment_1a(number_of_simulations, n, current_max, resistance_actual))
        experiments[-1].run_experiment()

    if not len(experiments) == 4:
        raise ValueError("Number of experiments in Exercise 1.a does not equal 4. Are the correct values for 'iterations' set?\nIterations should equal [10, 100, 1000, 10000]")

    if show_plot:
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.suptitle("Figure 1-1: Estimated resustance values for various sample sizes.")
        fig.supxlabel("Experiment Number")
        fig.supylabel("Resistance Estimate (Ohms)")

        axs[0, 0].plot(experiments[0].get_regression_values(), '.')
        axs[0, 0].axhline(y = experiments[0].get_resistance_actual())
        axs[0, 0].set_title("N = 10")

        axs[0, 1].plot(experiments[1].get_regression_values(), '.')
        axs[0, 1].axhline(y = experiments[1].get_resistance_actual())
        axs[0, 1].set_title("N = 100")

        axs[1, 0].plot(experiments[2].get_regression_values(), '.')
        axs[1, 0].axhline(y = experiments[2].get_resistance_actual())
        axs[1, 0].set_title("N = 1,000")

        axs[1, 1].plot(experiments[3].get_regression_values(), '.')
        axs[1, 1].axhline(y = experiments[3].get_resistance_actual())
        axs[1, 1].set_title("N = 10,000")

        print("Close plot to continue.")
        print()

        plt.show()

    return 0
    

def run_exercise_1b(number_of_simulations=1000, show_plot=True):
    print("Running Experiment 1b...")
    np.random.seed(27)

    iterations = [10, 100, 1000, 10000]
    current_actual = 0.01           # Amps
    resistance_actual = 1000        # Ohms

    experiments = []
    sigmas = []
    

    for n in iterations:
        experiments.append(ResistorExperiment_1b(number_of_simulations, n, current_actual, resistance_actual))
        experiments[-1].run_experiment()
        sigmas.append(experiments[-1].get_standard_deviation())
    
    if show_plot:
        fig, ax = plt.subplots()
        fig.suptitle("Figure 1-2: Standard deviation R(N)")
        fig.supxlabel("N")
        fig.supylabel("std Rest")

        ax.loglog(iterations, sigmas, 'o')

        x_values= np.arange(iterations[0], iterations[-1], 10)
        y_theoretical = 1 / (x_values**(1/2) * current_actual)
        ax.loglog(x_values, y_theoretical, '--', color='tab:blue')

        print("Close plot to continue.")
        print()
        
        plt.show()

    return 0


class ResistorExperiment_1a ( ResistorExperiment ):
    def __init__(self, simulations, iterations, current_max, resistance_actual):
        super().__init__(simulations, iterations, current_max, resistance_actual)


    def generate_current_samples(self):
        return np.random.uniform(-self.i_max, self.i_max, self.iters)


    def generate_noise_samples(self):
        return np.random.normal(0, 1, self.iters)


    def generate_voltage_samples(self, i_samples, n_samples):
        return np.array([self.r_act * i + n for i, n in zip(i_samples, n_samples)])


class ResistorExperiment_1b ( ResistorExperiment ):
    def __init__(self, simulations, iterations, current_max, resistance_actual):
        super().__init__(simulations, iterations, current_max, resistance_actual)


    def generate_current_samples(self):
        return self.i_max * np.ones(self.iters)


    def generate_noise_samples(self):
        return np.random.normal(0, 1, self.iters)


    def generate_voltage_samples(self, i_samples, n_samples):
        return np.array([self.r_act * i + n for i, n in zip(i_samples, n_samples)])


if __name__ == "__main__":
    main()
