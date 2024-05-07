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
            
def fourier_series_wave(num_harmonics, frequency=1, smoothness=1.):
    """
    Generates a Fourier series function with given number of harmonics.
    Amplitudes decrease with frequency and the total RMS is normalized 
    to sqrt(2)/2 like a normal sine wave.
    """
    def damped_fourier_wave_func(n_cycles):
        result = np.zeros_like(n_cycles)
        total_power = 0
        amplitudes = []
        phases = []
        order = smoothness

        # Generate harmonics with random amplitudes influenced by harmonic number
        for n in range(1, num_harmonics + 1):
            # Higher order harmonics have lower amplitudes
            amplitude = np.random.normal(loc=1.0/(n**order), scale=0.5/(n**order))  
            amplitude = abs(amplitude)  # Ensure non-negative amplitude
            phase = np.random.uniform(0, 2 * np.pi)
            amplitudes.append(amplitude)
            phases.append(phase)

            # Adding the harmonic to the series
            result += amplitude * np.sin(2 * np.pi * frequency * n * n_cycles + phase)
            total_power += amplitude**2  # Power of each component, RMS^2

        # Normalize the total RMS of the series to be 1
        rms_scale = np.sqrt(1 / total_power)
        normalized_result = result * rms_scale

        return normalized_result

    return damped_fourier_wave_func

def noisy_sine_wave(frequency=1, smoothing=20, noise_scale=0.1):
    # Noisy sine
    def noisy_sin_wave_func(n_cycles):
        # Deterministic sinusoidal component
        deterministic = np.sin(2 * np.pi * frequency * n_cycles)
        
        # Generate random noise
        random_noise = np.random.normal(scale=noise_scale, size=n_cycles.shape)
        
        # Smooth the random noise
        smoothed_noise = gaussian_filter1d(random_noise, sigma=smoothing)
        
        # Combine the deterministic and random parts
        return deterministic + smoothed_noise

    return noisy_sin_wave_func

def parse_bounds(bounds):
    """ Parse the bounds input which can be a single tuple or a tuple of three tuples. """
    if isinstance(bounds[0], (int, float)):  # Single tuple for all dimensions
        return (bounds, bounds, bounds)
    return bounds  # Tuple of three tuples

def random_point_in_ellipsoid(bounds):
    """ Generate a random point within an ellipsoid defined by bounds on x, y, z axes. """
    
    # Parse bounds and calculate centers and radii
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = parse_bounds(bounds)
    x_radius = (x_max - x_min) / 2
    y_radius = (y_max - y_min) / 2
    z_radius = (z_max - z_min) / 2
    center_x = x_min + x_radius
    center_y = y_min + y_radius
    center_z = z_min + z_radius
    
    # Random angles and radius for a unit sphere
    phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
    theta = np.random.uniform(0, np.pi)    # Polar angle
    r = np.random.uniform(0, 1)            # Radius
    
    # Random point in unit sphere scaled to fit the ellipsoid
    x = r * np.sin(theta) * np.cos(phi) * x_radius + center_x
    y = r * np.sin(theta) * np.sin(phi) * y_radius + center_y
    z = r * np.cos(theta) * z_radius + center_z
    
    return x, y, z

