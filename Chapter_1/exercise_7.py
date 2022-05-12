import numpy as np


def main():
    print_header()


def print_header():
    pass


class Data:
    def __init__(self, name, inputs, measurements):
        self.name = name
        self.inputs = inputs
        self.measurements = measurements


    def get_inputs(self):
        return self.inputs


    def get_measurements(self):
        return self.measurements


    def get_name(self):
        return self.name


class Experiment:
    def __init__(self, a, samples, variance, simulations):
        self.a = a
        self.samples = samples
        self.variance = variance 
        self.simulations = simulations  
        self.data_list = []  
        self.estimates_lsr = []
        self.estimates_wls = []  

    
    def generate_inputs(self):
        raise NotImplementedError("Subclass should implement this.")


    def generate_measurements(self):
        raise NotImplementedError("Subclass should implement this.")

    
    def generate_weights(self):
        raise NotImplementedError("Subclass should implement this.")

    
    def generate_K_matrix(self, inputs):
        raise NotImplementedError("TODO: ths needs to be written")


    def run_simulation(self, sim_number):
        inputs = self.generate_inputs()
        measurements = self.generate_measurements(inputs)

        return Data(sim_number, inputs, measurements)
    

    def run_experiment(self):

        self.data_list = []

        for sim_number in range(0, self.simulations):
            sim_data = self.run_simulation(sim_number)
            self.data_list.append(sim_data)

            # TODO: if inputs are always the same, can this just be computed once?
            K = self.generate_K_matrix(sim_data.get_inputs())
            W = self.generate_weights()

            self.estimates_lsr.append(self.compute_least_squares(K, sim_data.get_measurements(), False))
            self.estimates_wls.append(self.compute_weighted_least_squares(K, sim_data.get_measurements(), W, False))
            # TODO: run WLS

    
    def compute_least_squares(K, y, stable=True):

        if not stable:
            return np.linalg.inv(K.T @ K) @ K.T @ y 

        raise NotImplementedError("Need to figure out the stable vs unstable methods in numpy.")


    def compute_weighted_least_squares(K, y, W, stable=True):
        
        if not stable:
            return np.linalg.inv(K.T @ W @ K) @ K.T @ W @ y 

        raise NotImplementedError("Need to figure out the stable vs unstable methods in numpy.")
            

class ExperimentA ( Experiment ):
    def __init__(self, a, samples, variance, simulations):
        super().__init__(a, samples, variance, simulations)


    def generate_inputs(self):
        return np.linspace(-3, 3, self.samples)


    def generate_measurements(self, inputs):
        noise = np.random.normal(0, self.variance, self.iters)
        return self.a * inputs + noise

    
    def generate_weights(self):
        return np.identity(self.samples)


class ExperimentB ( Experiment ):
    def __init__(self, a, samples, variance, simulations):
        super().__init__(a, samples, variance, simulations)


    def generate_inputs(self):
        return np.linspace(2, 5, self.samples)


    def generate_measurements(self, inputs):
        noise = np.random.normal(0, self.variance, self.iters)
        return self.a * inputs + noise


    def generate_weights(self):
        return np.identity(self.samples)


if __name__ == "__main__":
    main()
