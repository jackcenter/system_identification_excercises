class ResistorData:
    def __init__(self, current_samples, noise_samples, voltage_samples, regression_value):
        self.i_samples = current_samples
        self.n_samples = noise_samples
        self.v_samples = voltage_samples
        self.r_value = regression_value
