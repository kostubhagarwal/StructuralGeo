import numpy as np
import itertools
from .util import rotate, slip_normal_vectors

import logging
# Set up a simple logger
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('Geo')
log.setLevel(logging.DEBUG)
# logging.disable()

class GeoModel:
    """A 3D geological model that can be built up from geological processes.
    
    Parameters:
    - bounds: Tuple defining the model bounds in the form, if only one tuple is provided, 
            it is assumed to be the same for all dimensions. 
            (allmin, allmax) or ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    - resolution: Number of divisions in each dimension
    - dtype: Data type for the model data array 
    """
    EMPTY_VALUE = -1
    
    def __init__(self,  bounds=(0, 16), resolution=64, dtype = np.float32):
        self.resolution = resolution
        self.dtype = dtype
        self.bounds = bounds
        self.history = []
        self.snapshots = np.empty((0, 0, 0, 0))
        self.data = np.empty(0)
        self.xyz  = np.empty((0,0))
        self.X    = np.empty((0,0,0))
        self.Y    = np.empty((0,0,0))
        self.Z    = np.empty((0,0,0))
        
    def setup_mesh(self, bounds):
        """Sets up the 3D meshgrid based on given bounds."""
        # Check if bounds is a tuple of tuples or a single tuple
        if isinstance(bounds[0], tuple):
            # Assume bounds are provided as ((xmin, xmax), (ymin, ymax), (zmin, zmax))
            x_bounds, y_bounds, z_bounds = bounds
        else:
            # Apply the same bounds to x, y, and z if only one tuple is provided
            x_bounds = y_bounds = z_bounds = bounds

        # Create linspace for x, y, z
        x = np.linspace(*x_bounds, num=self.resolution, dtype=self.dtype)
        y = np.linspace(*y_bounds, num=self.resolution, dtype=self.dtype)
        z = np.linspace(*z_bounds, num=self.resolution, dtype=self.dtype)
        
        # Init 3D meshgrid for X, Y, and Z coordinates of the view field
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Combine flattened arrays into a 2D numpy array where each row is an (x, y, z) coordinate
        self.xyz = np.column_stack((self.X.flatten(), self.Y.flatten(), self.Z.flatten()))
        
        # Initialize data array with NaNs
        self.data = np.full(self.xyz.shape[0], np.nan, dtype=self.dtype)  
        
    def add_history(self, history):
        """Add one or more geological processes to model history.
        
        Parameters:
            history (GeoProcess or list of GeoProcess): A GeoProcess instance or a list of GeoProcess instances to be added to the model's history.
        """
        # Check if the input is not a list, make it a list
        if not isinstance(history, list):
            history = [history]  # Convert a single GeoProcess instance into a list

        # Check if all elements in the list are instances of GeoProcess
        if not all(isinstance(event, GeoProcess) for event in history):
            raise TypeError("All items in the history list must be instances of the GeoProcess class.")
        
        # Extend the existing history with the new history
        self.history.extend(history)
            
    def clear_history(self):
        """Clear all geological process from history."""
        self.history = []  
        
    def clear_data(self):
        """ Clear the model but retain build parameters."""
        self.snapshots = np.empty((0, 0, 0, 0))
        self.data = np.empty(0)
        self.xyz  = np.empty((0,0))
        self.X    = np.empty((0,0,0))
        self.Y    = np.empty((0,0,0))
        self.Z    = np.empty((0,0,0))                  

    def compute_model(self):
        """Compute the present day model based on the geological history
        
        Snapshots:
        The xyz mesh is saved to a preallocated array for use in the forward pass.
        The starting state [0] is always required, additional snapshots are needed 
        at the start of any deposition process. 
                
        Backward pass:
        The xyz mesh is backtracked through history using the transformations
        stored in the history list. Intermediate states are stored at required
        intervals for the forward pass.
        
        Forward pass:
        The deposition events are applied to the xyz mesh in its intermediate 
        transformed state.        
        """
        if len(self.history) == 0:
            raise ValueError("No geological history to compute.")
        
        self.setup_mesh(self.bounds)        
        # Determine how many snapshots are needed for memory pre-allocation
        self.snapshot_indices = self._prepare_snapshots()
        self.snapshots = np.empty((len(self.snapshot_indices), *self.xyz.shape))
        log.info(f"Intermediate mesh states will be saved at {self.snapshot_indices}")
        log.info(f"Total gigabytes of memory required: {self.snapshots.nbytes * 1e-9:.2f}")
    
        # Backward pass to reverse mesh grid of points
        self._backward_pass()
        
        # Forward pass to apply deposition events
        self._forward_pass()
          
    def _prepare_snapshots(self):
        """ Determine when to take snapshots of the mesh during the backward pass.
        
        Snapshots of the xyz mesh should be taken at end of a transformation sequence
        """  
        # Always include the time start state of mesh      
        snapshot_indices = [0]
        for i in range(1,len(self.history)):
            if isinstance(self.history[i], Deposition) and isinstance(self.history[i-1], Transformation):
                snapshot_indices.append(i)
        
        return snapshot_indices

    def _backward_pass(self):
        """ Backtrack the xyz mesh through the geological history using transformations."""
        # Make a copy of the model xyz mesh to apply transformations
        current_xyz = self.xyz.copy()
        
        i = len(self.history) - 1
        for event in reversed(self.history):
            # Store snapshots of the mesh at required intervals
            if i in self.snapshot_indices:
                # The final state (index 0) uses the actual xyz since no further modifications are made,
                # avoiding unnecessary copying for efficiency.
                if i != 0:
                    self.snapshots[self.snapshot_indices.index(i)] = np.copy(current_xyz)
                else:
                    self.snapshots[0] = current_xyz 
                       
                log.debug(f"Snapshot taken at index {i}")    
            # Apply transformation to the mesh (skipping depositon events that do not alter the mesh)    
            if isinstance(event, Transformation):
                current_xyz, _ = event.run(current_xyz, self.data)
            i -= 1
            
    def _forward_pass(self):
        """ Apply deposition events to the mesh based on the geological history."""
        for i, event in enumerate(self.history):
            # Update mesh coordinates as required by fetching snapshot from the backward pass
            if i in self.snapshot_indices:
                snapshot_index = self.snapshot_indices.index(i)
                current_xyz = self.snapshots[snapshot_index,...]
            if isinstance(event, Deposition):
                _, self.data = event.run(current_xyz, self.data)
        
    def fill_nans(self, value = EMPTY_VALUE):
        assert self.data is not None, "Data array is empty."
        indnan = np.isnan(self.data)
        self.data[indnan] = value
        return self.data  

class GeoProcess:
    """Base class for all geological processes.
    
    Conventions:
    Strike, dip, and rake are in degrees.    
    """    
    pass

class Deposition(GeoProcess):
    """
    Base class for all deposition processes, such as layers and dikes.
    
    Depositions modify the geological data (e.g., rock types) associated with a mesh point
    without altering or transforming the mesh.
    """
    def run(self, xyz, data):
        raise NotImplementedError()
    
class Transformation(GeoProcess):
    """
    Base class for all transformation processes, such as tilting and folding.
    
    Transformations modify the mesh coordinates without altering the data contained at each point.
    Frame convention is that north is strike on positive x-axis
    """
    def run(self, xyz, data):
        raise NotImplementedError()
        
class Layer(Deposition):
    def __init__(self, base, width, value):
        self.base = base
        self.width = width
        self.value = value

    def run(self, xyz, data):
        # Extract z-coordinates
        z_coords = xyz[:, 2]

        # Create a mask where z is within the specified range and data is None
        mask = (self.base <= z_coords) & (z_coords <= self.base + self.width) & (np.isnan(data))

        # Apply the mask and update data where condition is met
        data[mask] = self.value

        # Return the unchanged xyz and the potentially modified data
        return xyz, data
    
class Bedrock(Deposition):
    """ Fill the model with a base level of bedrock.
    
    Parameters:
    - base: The z-coordinate of the base level (top of the rock layer)
    - value: The rock type value to assign to the bedrock layer
    """
    def __init__(self, base, value):
        self.base = base
        self.value = value

    def run(self, xyz, data):
        # Extract z-coordinates
        z_coords = xyz[:, 2]

        # Create a mask where z is below the base level
        mask = z_coords <= self.base

        # Apply the mask and update data where condition is met
        data[mask] = self.value

        # Return the unchanged xyz and the potentially modified data
        return xyz, data
    
class Sedimentation(Deposition):
    """ Fill with layers of sediment, with thickness given by a random variable.
    
        Parameters:
              - height: the total height of the sedimentary sequence
              - value_list: a list of values to sample from for rock types
              - value_selector: an optional object to sample values from the list
              - thickness_callable: a random number generator to use for thicknesses

    """
    def __init__(self, height, value_list, value_selector=None, thickness_callable=None, ):
        self.height = height
        self.value_list = value_list
        # Set up the value generator which returns next value from the list
        if  value_selector is None:
            # Default is to cycle through the list
            self.value_selector = itertools.cycle(value_list) 
        else:
            # User provided value selector (must be an iterator that accepts a list)
            self.value_selector = value_selector(value_list)      
        
        # Initialize the thickness function, default to constant thickness of 1
        self.thickness_callable = thickness_callable if thickness_callable else self.thickness_callable_default
        self.values_sequence_used = []
        self.thickness_sequence_used = []
        self.rebuild = False
        
    def thickness_callable_default(self):
        return 1.0

    def run(self, xyz, data):
        if self.rebuild:
            return self.rebuild_sequence(xyz, data)
        else:
            return self.generate_sequence(xyz, data)

    def generate_sequence(self, xyz, data):
        # Get the lowest mesh point to build layers from the bottom up        
        current_base = self.get_lowest_nan(xyz, data)
        
        # Build up until the total height is reached
        while current_base < self.height:
            # Sample the next sediment value
            value = next(self.value_selector)
            self.values_sequence_used.append(value)
            # Sample next layer thickness
            layer_thickness = self.thickness_callable()
            self.thickness_sequence_used.append(layer_thickness)
            # Do not exceed the total height
            current_top = min(current_base + layer_thickness, self.height)
            
            # Mask for the current layer
            mask = (xyz[:, 2] < current_top) & (xyz[:, 2] >= current_base) & (np.isnan(data))
            
            # Assign the current sediment value to the layer
            if np.any(mask):
                data[mask] = value
            
            # Update the base for the next layer
            current_base = current_top   
            
        # Flag to rebuild the sequence if needed
        self.rebuild = True
             
        return xyz, data 

    def rebuild_sequence(self, xyz, data):
        # Get the lowest mesh point to build layers from the bottom up        
        current_base = self.get_lowest_nan(xyz, data)
        
        # Build up until the total height is reached
        for val, thickness in zip(self.values_sequence_used, self.thickness_sequence_used):
            # Do not exceed the total height
            current_top = min(current_base + thickness, self.height)
            
            # Mask for the current layer
            mask = (xyz[:, 2] < current_top) & (xyz[:, 2] >= current_base) & (np.isnan(data))
            
            # Assign the current sediment value to the layer
            if np.any(mask):
                data[mask] = val
            
            # Update the base for the next layer
            current_base = current_top
        return xyz, data
            
    def get_lowest_nan(self, xyz, data):
        # Get the lowest mesh point with NaN in data to build layers from the bottom up
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            lowest = np.min(xyz[nan_mask, 2])
            return lowest
        else:
            log.warning("No NaN values found in data; no layers will be added.")
            return float('Inf')  # Return early if there are no NaN values to process   

class Dike(Deposition):
    """ Insert a dike to overwrite existing data values.
    
    Parameters:
    - strike: Strike angle in CW degrees (center-line of the dike) from north
    - dip: Dip angle in degrees
    - width: Net Width of the dike
    - origin: Origin point of the local coordinate frame
    - value: Value of rock-type to assign to the dike 
    """
    def __init__(self, strike=0., dip=90., width=1., origin=(0,0,0), value=0.):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.width = width
        self.origin = np.array(origin)
        self.value = value

    def run(self, xyz, data):
        # Calculate rotation matrices
        M1 = rotate([0, 0, 1.], -self.strike)  # Rotation around z-axis for strike
        M2 = rotate([0., 1., 0], self.dip)     # Rotation around y-axis in strike frame for dip

        N = M1 @ M2 @ [0.0, 0.0, 1.0]
        N /= np.linalg.norm(N)  # Normalize the normal vector

        # Calculate distances from the dike plane
        dists = np.dot(xyz - self.origin, N)

        # Update data based on whether points are within the width of the dike and only where there
        # is existing data to avoid sky dikes
        mask = (np.abs(dists) <= self.width / 2.0) & (~np.isnan(data))
        
        data[mask] = self.value

        return xyz, data
    
class ErosionLayer(Deposition):
    """ Erode down to some depth from the peak of the surface.
    
    Parameters:
    - thickness: Thickness of the erosion layer, measured from the peak of the surface
    """
    
    def __init__(self, thickness):
        self.thickness = thickness
        self.peak = None
        self.value = np.nan
        
    def run(self, xyz, data):
        # Get the highest z-coordinate that is not NaN
        xyz_valued = xyz[~np.isnan(data)]
        self.peak = np.max(xyz_valued[:, 2])
        
        # Mask for points below the peak minus the erosion depth
        mask = xyz[:, 2] > self.peak - self.thickness
        
        # Apply the mask and update data where condition is met
        data[mask] = self.value

        # Return the unchanged xyz and the potentially modified data
        return xyz, data

class Tilt(Transformation):
    """ Tilt the model by a given strike and dip and an origin point.
    
    Parameters:
    - strike: Strike angle in CW degrees (center-line of the dike) from north
    - dip: Dip of the tilt in degrees (CW from the strike axis)
    - origin: Origin point for the tilt
    """
    def __init__(self, strike, dip, origin=(0,0,0)):
        self.strike = np.radians(strike)  # Convert degrees to radians
        self.dip = np.radians(dip)  # Convert degrees to radians
        self.origin = np.array(origin)

    def run(self, xyz, data):
        # Calculate rotation axis from strike (rotation around z-axis)
        axis = rotate([0, 0, 1], -self.strike) @ [0, 1., 0]
        # Calculate rotation matrix from dip (tilt)
        R = rotate(axis, -self.dip)
        
        # Apply rotation about origin -> translate to origin, rotate, translate back
        # Y = R * (X - O) + O
        xyz = xyz @ R.T + (-self.origin @ R.T + self.origin)

        # Apply rotation to xyz points
        return xyz, data  # Assuming xyz is an Nx3 numpy array 
      
class Fold(Transformation):
    """ Generalized fold transformation. 
    Convention is that 0 rake with 90 dip fold creates vertical folds.
    
    Parameters:
    - strike: Strike in degrees
    - dip: Dip in degrees
    - rake: Rake in degrees
    - period: Period of the fold in units of the mesh
    - amplitude: Amplitude of the fold (motion along slip_vector)
    - shape: Shape parameter for the fold
    - origin: Origin point for the fold
    - periodic_func: Custom periodic function for the fold (replaces default cosine function)
                    User provided function should be 1D and accept an array of n_cycles
    """
    
    def __init__(self, strike = 0., 
                 dip = 90., rake = 0., 
                 period = 50., 
                 amplitude = 10., 
                 shape = 0.0, 
                 origin=(0, 0, 0),
                 periodic_func=None):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.rake = np.radians(rake)
        self.period = period
        self.amplitude = amplitude
        self.shape = shape
        self.origin = np.array(origin)
        # Accept a custom periodic function or use the default otherwise
        self.periodic_func = periodic_func if periodic_func else self.periodic_func_default

    def run(self, xyz, data):
        # Adjust the rake to have slip vector (fold amplitude) perpendicular to the strike
        slip_vector, normal_vector = slip_normal_vectors(self.rake + np.pi/2, self.dip, self.strike)

        # Translate points to origin coordinate frame
        v0 = xyz - self.origin
        # Orthogonal distance from origin along U
        fU = np.dot(v0, normal_vector)
        # Calculate the number of cycles orthogonal distance
        n_cycles =  fU / self.period
        # Get the displacement as a function for n_cycles
        displacement_distance = self.amplitude * self.periodic_func(n_cycles)      
        # Calculate total displacement for each point, recast off as a column vector
        displacement_vector = slip_vector * displacement_distance[:, np.newaxis] 
        # Return to global coordinates
        xyz_transformed = xyz + displacement_vector + self.origin

        return xyz_transformed, data  
    
    def periodic_func_default(self, n_cycles):
        # Normalize to amplitude of 1
        norm = (1 + self.shape**2)**0.5
        func = np.cos(2 * np.pi * n_cycles) + self.shape * np.cos(3 * 2 * np.pi * n_cycles)  
        return func / norm   

class Slip(Transformation):
    """Gereralized slip transformation.
    
    Parameters:
    - displacement_func: Custom displacement function for the slip. Function should map 
    a distance from the slip plane to a displacement value.
    - strike: Strike in degrees
    - dip: Dip in degrees
    - rake: Rake in degrees
    - amplitude: Amplitude of the slip (motion along slip_vector)
    - origin: Origin point for the slip (local coordinate frame)
    
    """
    def __init__(self,                 
                displacement_func,
                strike = 0., 
                dip = 90., 
                rake = 0., 
                amplitude = 2., 
                origin=(0, 0, 0),
                ):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.rake = np.radians(rake)
        self.amplitude = amplitude
        self.origin = np.array(origin)
        self.displacement_func = displacement_func
    
    def default_displacement_func(self, distances):
        # A simple linear displacement function as an example
        return np.zeros(np.shape(distances))  # Displaces positively where the distance is positive

    def run(self, xyz, array):
        # Slip is measured from dip vector, while the slip_normal convention is from strike vector, add 90 degrees
        slip_vector, normal_vector = slip_normal_vectors(self.rake , self.dip, self.strike)
        
        # Translate points to origin coordinate frame
        v0 = xyz - self.origin        
        # Orthogonal distance from origin along U
        distance_to_slip = np.dot(v0, normal_vector)
        # Apply the displacement function to the distances along normal
        displacements = self.amplitude * self.displacement_func(distance_to_slip)
        # Calculate the movement vector along slip direction
        displacement_vectors = displacements[:, np.newaxis] * slip_vector
        # Return to global coordinates and apply the displacement
        xyz_transformed = xyz + displacement_vectors + self.origin
        return xyz_transformed, array
    
class Fault(Slip):
    """
    A subclass of Slip specifically for modeling brittle fault transformations where
    displacement occurs as a sharp step function across the fault plane. This class
    implements a binary displacement function that represents the sudden shift
    characteristic of brittle faults.

    Parameters:
    - strike (float): Strike angle in degrees
    - dip (float): Dip angle in degrees
    - rake (float): Rake angle in degrees, convention is 0 for side-to-side motion
    - amplitude (float): The maximum displacement magnitude along the slip_vector.
    - origin (tuple of float): The x, y, z coordinates from which the fault originates within the local coordinate frame.
    
    This implementation causes a displacement strictly on one side of the fault, making it suitable for
    simulating scenarios where a clear delineation between displaced and stationary geological strata is necessary.
    
    Example:
        # Creating a Fault instance with specific geological parameters
        fault = Fault(strike=30, dip=60, rake=90, amplitude=5, origin=(0, 0, 0))
    """
    def __init__(self, 
                strike = 0., 
                dip = 90., 
                rake = 0., 
                amplitude = 2., 
                origin=(0, 0, 0),
                ):
        super().__init__(self.fault_displacement, strike, dip, rake, amplitude, origin)
        self.rotation = 0
        
    def fault_displacement(self, distances):
        return self.amplitude * np.sign(distances)
    
class Shear(Slip):
    """ A subclass of Slip for modeling shear transformations, a plastic deformation process.
    Displacement is modeled as a sigmoid function that increases with distance from the slip plane.
    
    Parameters:
    - strike (float): Strike angle in degrees
    - dip (float): Dip angle in degrees
    - rake (float): Rake angle in degrees, convention is 0 for side-to-side motion
    - amplitude (float): The maximum displacement magnitude along the slip_vector.
    - origin (tuple of float): The x, y, z coordinates from which the fault originates within the local coordinate frame.
    - steepness (float): The steepness of the sigmoid function, controlling the rate of change of displacement.
    """
    def __init__(self, 
                strike = 0., 
                dip = 90., 
                rake = 0., 
                amplitude = 2., 
                steepness = 1.,
                origin=(0, 0, 0),
                ):
        self.steepness = steepness
        super().__init__(self.shear_displacement, strike, dip, rake, amplitude, origin)
        
    def shear_displacement(self, distances):
        # The sigmoid function will be centered around zero and will scale with amplitude
        return self.amplitude * (1 / (1 + np.exp(-self.steepness * distances)))
    
    def run(self, xyz, array):
        # Apply the shear transformation
        xyz_transformed, array = super().run(xyz, array)
        return xyz_transformed, array