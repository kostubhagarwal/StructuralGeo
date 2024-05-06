import numpy as np
from scipy.ndimage import gaussian_filter1d
import itertools

class NonRepeatingRandomListSelector:
    """ Generate unique non-repeating random elements from a list """
    def __init__(self, elements):
        # Attempt to convert input to a list
        try:
            self.full_range = list(elements)
        except TypeError:
            raise ValueError("Input must be an iterable that can be converted to a list")               
        self.num_samples = len(self.full_range)
        self.previous_sample_index = None
    
    def __next__(self):
        if self.num_samples == 1:
            # Only one sample in the list, not possible to non-repeat
            return self.full_range[0]
        
        # Assertion to avoid infinite loop
        assert(self.num_samples > 1) 
        while True:
            index = np.random.randint(0, self.num_samples)
            if index != self.previous_sample_index:  # Ensure it's not the previous index
                self.previous_sample_index = index
                return self.full_range[index]

def noisy_sine_wave(frequency=1, smoothing=20, noise_scale=0.1):
    # Noisy sine
    def customized_wave_func(n_cycles):
        # Deterministic sinusoidal component
        deterministic = np.sin(2 * np.pi * frequency * n_cycles)
        
        # Generate random noise
        random_noise = np.random.normal(scale=noise_scale, size=n_cycles.shape)
        
        # Smooth the random noise
        smoothed_noise = gaussian_filter1d(random_noise, sigma=smoothing)
        
        # Combine the deterministic and random parts
        return deterministic + smoothed_noise

    return customized_wave_func