import numpy as np
from .geo import Layer, Tilt, Fold, Dike, Slip

class GeoHistory:
    def __init__(self):
        self.transformations = []

    def add_transformation(self, transformation):
        self.transformations.append(transformation)

    def generate_history(self):
        raise NotImplementedError("Subclasses should implement specific history generation methods.")
    

class SedimentaryHistory(GeoHistory):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def generate_history(self):
        # Generate base layers with sequential values
        base = self.config['base_layers']['base_init']
        base_layers = np.random.lognormal(self.config['mean_log'], self.config['sigma_log'])
        for count in enumerate(base_layers):
            width = np.random.normal(self.config['mean_width'], self.config['sigma_width'])
            value = np.random.choice(value)
            self.add_transformation(Layer(base, width, value))
            
        
        



