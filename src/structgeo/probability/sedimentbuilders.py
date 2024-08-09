import numpy as np

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
        
        # Log normal distribution parameters https://en.wikipedia.org/wiki/Log-normal_distribution
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
    
class MarkovSedimentHelper:
    """Helper class for handling Markov process logic in sediment generation.
    
    Parameters:
        - categories (list): List of sediment categories.
        - rng (np.random.Generator): Random number generator.
        - thickness_bounds (tuple): Bounds for min/max allowable layer thicknesses.
    """
    
    def __init__(self, categories, rng, thickness_bounds=(100, 1000), thickness_variance=0.1, dirichlet_alpha=0.8):
        self.cats = categories
        self.rng = rng
        self.thickness_bounds = thickness_bounds
        self.thickness_variance = thickness_variance
        self.transition_matrix = self.randomize_transition_matrix(dirichlet_alpha)

    def randomize_transition_matrix(self, alpha):
        """Randomize the Markov transition matrix for sediment categories."""
        transition_matrix = {}
        for val in self.cats:
            probabilities = self.rng.dirichlet(alpha * np.ones(len(self.cats)))
            transition_matrix[val] = probabilities
        return transition_matrix

    def next_layer_category(self, current_val):
        """Determine the next layer category based on the current category and Markov process."""
        if current_val is None:
            return np.random.choice(self.cats)
        else:
            return np.random.choice(self.cats, p=self.transition_matrix[current_val])

    def next_layer_thickness(self, current_thick):
        """Determine the next layer thickness based on the current thickness."""
        if current_thick is None:
            return self.rng.uniform(*self.thickness_bounds)
        else:
            next_thick = current_thick * np.random.normal(1, self.thickness_variance)  # Induce some variation
            next_thick = np.clip(next_thick, *self.thickness_bounds)  # Bound thicknesses
            return next_thick