import matplotlib.pyplot as plt
import numpy as np

from Chapter_1.ResistorExperiment import ResistorExperiment


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


def run_exercise_2(number_of_simulations=100000, show_plot=True):
    print("Running Experiment 2...")
    np.random.seed(2)

    iterations = [1, 2, 4, 8]
    current_actual = 0.01           # Amps
    resistance_actual = 1000        # Ohms

    experiments_normal = []
    experiments_uniform = []

    for n in iterations:
        experiments_normal.append(Experiment_2a(number_of_simulations, n, current_actual, resistance_actual))
        experiments_normal[-1].run_experiment()

        experiments_uniform.append(Experiment_2b(number_of_simulations, n, current_actual, resistance_actual))
        experiments_uniform[-1].run_experiment()

    if show_plot:
        fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
        fig.suptitle("Figure 1-3: Evolution of the pdf of R as a function of N")
        fig.supxlabel("R")

        for i in range(0, 4):
            axs[i].hist(experiments_normal[i].get_regression_values(), bins=100, density=True, histtype='step', label='Normal')
            axs[i].hist(experiments_uniform[i].get_regression_values(), bins=100, density=True, histtype='step', label='Uniform')
            axs[i].set_ylabel("N = " + str(experiments_normal[i].iters))

        axs[0].legend()

        print("Close plot to continue.")
        print()
        
        plt.show()

    return 0


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
