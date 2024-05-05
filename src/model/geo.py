import numpy as np
import itertools
from .util import rotate

import logging
# Set up a simple logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('Geo')
log.setLevel(logging.DEBUG)

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
        
        self.setup_mesh(bounds)  
        
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
        """Add one or more geological processes to model history."""
        if not all(isinstance(event, GeoProcess) for event in history):
            raise TypeError("All items in the history list must be instances of the GeoProcess class.")                
        else:
            self.history.extend(history)
            
    def clear_history(self):
        """Clear all geological process from history."""
        self.history = []                

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
    pass

class Deposition(GeoProcess):
    """
    Base class for all deposition processes, such as layers and dikes.
    
    Depositions modify the geological data (e.g., rock types) associated with a mesh point
    without altering or transforming the mesh.
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
        self.thickness_callable = thickness_callable if thickness_callable else lambda: 1

    def run(self, xyz, data):
        # Get the lowest mesh point to build layers from the bottom up
        current_base = np.min(xyz[:, 2])
        
        # Build up until the total height is reached
        while current_base < self.height:
            # Sample the next sediment value
            value = next(self.value_selector)
            # Sample next layer thickness
            layer_thickness = self.thickness_callable()
            # Do not exceed the total height
            current_top = min(current_base + layer_thickness, self.height)
            
            # Mask for the current layer
            mask = (xyz[:, 2] < current_top) & (xyz[:, 2] >= current_base) & (np.isnan(data))
            
            # Assign the current sediment value to the layer
            if np.any(mask):
                data[mask] = value
            
            # Update the base for the next layer
            current_base = current_top
        
        return xyz, data    

class Dike(Deposition):
    
    def __init__(self, strike, dip, width, point, data_value):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.width = width
        self.point = np.array(point)
        self.data_value = data_value

    def run(self, xyz, data):
        # Calculate rotation matrices
        M1 = rotate([0, 0, 1], -self.strike)  # Rotation around z-axis for strike
        M2 = rotate([0, 1, 0], self.dip)      # Rotation around y-axis for dip

        # Normal vector of the dike plane
        N = M1 @ M2 @ [0.0, 0.0, 1.0]
        N /= np.linalg.norm(N)  # Normalize the normal vector

        # Calculate distances from the dike plane
        dists = np.dot(xyz - self.point, N)

        # Update data based on whether points are within the width of the dike and only where there
        # is existing data to avoid sky dikes
        mask = (np.abs(dists) <= self.width / 2.0) & (~np.isnan(data))
        
        data[mask] = self.data_value

        return xyz, data
    
class Transformation(GeoProcess):
    """
    Base class for all transformation processes, such as tilting and folding.
    
    Transformations modify the mesh coordinates without altering the data contained at each point.
    """
    def run(self, xyz, data):
        raise NotImplementedError("Transformation must implement 'run' method.")

class Tilt(Transformation):
    def __init__(self, strike, dip):
        self.strike = np.radians(strike)  # Convert degrees to radians
        self.dip = np.radians(dip)  # Convert degrees to radians

    def run(self, xyz, data):
        # Calculate rotation axis from strike (rotation around z-axis)
        axis = rotate([0, 0, 1], -self.strike) @ [0, 1, 0]

        # Calculate rotation matrix from dip (tilt)
        R = rotate(axis, -self.dip)

        # Apply rotation to xyz points
        return xyz @ R.T, data  # Assuming xyz is an Nx3 numpy array

class Fold(Transformation):
    def __init__(self, strike = 0, dip = 90, rake = 0, period = 50, amplitude = 10, shape = 0, offset=0, point=[0, 0, 0]):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.rake = np.radians(rake)
        self.period = period
        self.amplitude = amplitude
        self.shape = shape
        self.offset = offset
        self.point = np.array(point)

    def run(self, xyz, data):
        # Calculate rotation matrices
        M1 = rotate([0, 0, 1], -(self.rake + np.pi / 2))
        M2 = rotate([0, 1, 0], -(self.dip))
        M3 = rotate([0, 0, 1], -(self.strike))

        # Normal vector of the fold
        N = M3 @ M2 @ M1 @ [0., 1.0, 0.0]
        M1 = rotate([0, 0, 1], -(self.rake))
        V = M3 @ M2 @ M1 @ [0., 1.0, 0.0]
        U = np.cross(N[:3], V[:3])

        new_xyz = np.empty_like(xyz)
        for i, point in enumerate(xyz):
            v0 = point - self.point
            fU = np.dot(v0, U) / np.linalg.norm(U) - self.offset * self.period
            inside = 2 * np.pi * fU / self.period
            off = self.amplitude * (np.cos(inside) + self.shape * np.cos(3 * inside))
            T = N * off
            new_xyz[i] = point + T[:3]

        return new_xyz, data

class Shear(Transformation):
    def __init__(self, shear_amount, dims=[50,50,50]):
        self.shear_amount = shear_amount
        self.dims = dims
        
    def run(self, xyz, array):
        """
        Apply shear transformation to a 3D array along the x-axis.
        Parameters:
        - array: Input 3D array
        - shear_amount: Shear amount (fraction of array size)

        Returns:
        - Sheared 3D array
        """
        # Get array dimensions
        width, height, depth = self.dims

        array = array.reshape(self.dims)
        # Create sheared array
        sheared_array = np.zeros_like(array)

        # Apply shear transformation along x-axis
        for z in range(depth):
            for x in range(width):
                shear = int(self.shear_amount * x)
                sheared_array[x, :, z] = np.roll(array[x, :, z], shear)

        return xyz, sheared_array.flatten()