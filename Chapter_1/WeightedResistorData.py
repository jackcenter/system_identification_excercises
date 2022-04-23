import numpy as np

class WeightedResistorData:
    def __init__(self, current_samples, voltage_samples, weights, regression_value):
        self.i_samples = current_samples
        self.v_samples = voltage_samples
        self.weights = weights
        self.r_value = regression_value
        self.r_type = None


    def get_current_samples(self):
        return self.i_samples


    def get_voltage_samples(self):
        return self.v_samples

    
    def get_weights(self):
        return self.weights

    
    def get_regression_value(self):
        return self.regression_value


    def run_least_squares_regression(self):
        self.regression_value = np.dot(self.i_samples, self.v_samples) / np.dot(self.i_samples, self.i_samples)
        self.r_type = "Least Squares Regression"


    def run_weighted_least_squares(self):
        self.regression_value = np.sum(self.v_samples * self.i_samples / self.weights) / np.sum(self.i_samples * self.i_samples / self.weights)
        self.r_type = "Weighted Least Squares"


    @staticmethod
    def combine_two_data_sets(dataset_1, dataset_2):
        current_samples = np.concatenate([dataset_1.get_current_samples(), dataset_2.get_current_samples()])
        voltage_samples = np.concatenate([dataset_1.get_voltage_samples(), dataset_2.get_voltage_samples()])
        weights = np.concatenate([dataset_1.get_weights(), dataset_2.get_weights()])
        regression_value = None
        return WeightedResistorData(current_samples, voltage_samples, weights, regression_value)
