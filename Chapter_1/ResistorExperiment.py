import numpy as np
from ResistorData import ResistorData


class ResistorExperiment:
    def __init__(self, simulations, iterations, current_max, resistance_actual):
        self.sims = simulations
        self.iters = iterations
        self.i_max = current_max
        self.r_act = resistance_actual       

        self.results = []
        self.mean = None
        self.std_dev = None


    def get_regression_values(self):
        # TODO: check for empty results
        return np.array([datum.r_value for datum in self.results])
    

    def get_resistance_actual(self):
        return self.r_act


    def get_mean(self):
        if not self.mean:
            self.calculate_mean()

        return self.mean


    def get_standard_deviation(self):
        if not self.std_dev:
            self.calculate_standard_deviation()

        return self.std_dev

    
    def calculate_mean(self):
        self.mean = np.mean(self.get_regression_values())


    def calculate_standard_deviation(self):
        self.std_dev = np.std(self.get_regression_values())


    def generate_current_samples(self):
        raise NotImplementedError("Subclass should implement this.")


    def generate_noise_samples(self):
        raise NotImplementedError("Subclass should implement this.")


    def generate_voltage_samples(self, i_samples, n_samples):
        raise NotImplementedError("Subclass should implement this.")


    def run_simulation(self):

        i_samples = self.generate_current_samples()
        n_samples = self.generate_noise_samples()
        v_samples = self.generate_voltage_samples(i_samples, n_samples)
        r_value = self.calculate_least_squares_estimate(i_samples, v_samples)

        return ResistorData(i_samples, n_samples, v_samples, r_value)


    def run_experiment(self):
        
        self.results = []
        
        for _ in range(0, self.sims):
            self.results.append(self.run_simulation())

        self.calculate_mean()
        self.calculate_standard_deviation()

    @staticmethod
    def calculate_least_squares_estimate(i_samples, v_samples):
    
        return np.dot(i_samples, v_samples) / np.dot(i_samples, i_samples)
