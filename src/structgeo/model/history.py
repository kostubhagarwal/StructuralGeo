import numpy as np
from .geomodel import *
from .geoprocess import *

""" Collection of useful higher abstraction geo structures."""

class SedimentBuilder:
    def __init__(self, start_value, total_thickness, min_layers, max_layers, std=0.5):
        """
        Initialize the sediment builder to generate sediment layers with specific characteristics.

        Parameters:
        - start_value (int): The starting value for the sediment layers (e.g., rock type identifier).
        - total_thickness (float): The total desired thickness of all layers combined.
        - min_layers (int): Minimum number of layers.
        - max_layers (int): Maximum number of layers.
        """
        self.start_value = start_value
        self.total_thickness = total_thickness
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.std = std
        self.values, self.thicknesses = self.build_layers()

    def build_layers(self):
        n_layers = np.random.randint(self.min_layers, self.max_layers + 1)
        
        # desired mean and std of the layer thicknesses
        target_mean = self.total_thickness / n_layers
        target_std = self.std * target_mean
        
        # Log normal distribution parameters
        mu = np.log(target_mean**2/np.sqrt(target_mean**2 + target_std**2))
        sigma = np.sqrt(np.log(1 + target_std**2/target_mean**2))

        # Generate random thicknesses
        random_thicknesses = np.random.lognormal(mean=mu, sigma=sigma, size=n_layers)
        normalized_thicknesses = random_thicknesses / np.sum(random_thicknesses) * self.total_thickness
        values = [self.start_value + i for i in range(n_layers)]

        self.values, self.thicknesses = values, normalized_thicknesses

        return values, normalized_thicknesses

    def get_layers(self):
        """Returns a pair of lists containing the values and thicknesses of the layers."""
        return self.values, self.thicknesses

    def get_total_thickness(self):
        """Calculate and return the sum of the thicknesses of all layers."""
        return sum(self.thicknesses)
    
    def get_last_value(self):
        """Return the value of the last layer."""
        return self.values[-1]
    
class FaultSequence:
    """ Layout a sequence of faults from a starting point and direction"""

pass



