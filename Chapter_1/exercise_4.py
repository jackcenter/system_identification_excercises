import matplotlib.pyplot as plt
import numpy as np

from ResistorExperiment import ResistorExperiment
from ResistorData import ResistorData


def main():
     print_header()
     run_exercise_4()
     return 0


def print_header():
    print("=====================================================")
    print("                     Exercise 4:")
    print("             Importance of the choice of the") 
    print("              independent variable or input")
    print("=====================================================")
    print()


def run_exercise_4():
    print("Running Experiment 4...")
    np.random.seed(4)

    number_of_simulations = 100000
    iterations = [100]
    current_actual = 0.01           # Amps
    resistance_actual = 1000        # Ohms

    experiments_current = []        # current is the independent variable
    experiments_voltage = []        # voltage is the independent variable

    for n in iterations:
        experiments_current.append(ExperimentCurrent(number_of_simulations, n, current_actual, resistance_actual))
        experiments_current[-1].run_experiment()

        experiments_voltage.append(ExperimentVoltage(number_of_simulations, n, current_actual, resistance_actual))
        experiments_voltage[-1].run_experiment()

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.suptitle("Figure 1-5: Study of independent variable selection")
    fig.supxlabel("R")

    axs.hist(experiments_current[-1].get_regression_values(), bins=100, density=True, histtype='step')
    axs.hist(experiments_voltage[-1].get_regression_values(), bins=100, density=True, histtype='step')    

    plt.legend(["current", "voltage"])
    
    print("Close plot to continue.")
    print()

    plt.show()

class ExperimentCurrent ( ResistorExperiment ):

    def __init__(self, simulations, iterations, current_max, resistance_actual):
        super().__init__(simulations, iterations, current_max, resistance_actual)


    def generate_current_samples(self):
        return np.random.uniform(-self.i_max, self.i_max, self.iters)


    def generate_noise_samples(self):
        return np.random.normal(0, 1, self.iters)
    

    def generate_voltage_samples(self, i_samples, n_samples):
        return np.array([self.r_act * i + n for i, n in zip(i_samples, n_samples)])


class ExperimentVoltage ( ExperimentCurrent ):
    def __init__(self, simulations, iterations, current_max, resistance_actual):
        super().__init__(simulations, iterations, current_max, resistance_actual)

    
    @staticmethod
    def calculate_least_squares_estimate(i_samples, v_samples):
    
        return 1 / (np.dot(i_samples, v_samples) / np.dot(v_samples, v_samples))

if __name__ == "__main__":
    main()
