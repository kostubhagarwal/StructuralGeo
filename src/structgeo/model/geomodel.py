import numpy as np
from .geoprocess import *
from .util import rotate, slip_normal_vectors

import logging
# Set up a simple logger
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('Geo')
log.setLevel(logging.DEBUG)
logging.disable()

class GeoModel:
    """A 3D geological model that can be built up from geological processes.
    
    Parameters:
    bounds (Tuple): (allmin, allmax) or ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    resolution (int): Number of divisions in each dimension, 
                        or a tuple of 3 values for x, y, and z dimensions
    dtype (dtype): Data type for the model data array 
    """
    EMPTY_VALUE = -1
    
    def __init__(self,  bounds=(0, 16), resolution=128, dtype = np.float32):
        self.dtype = dtype
        self.bounds = bounds
        self.resolution = resolution
        self.history = []
        # Placeholders for mesh data
        self.data = np.empty(0) # Vector of data values on mesh points
        self.xyz  = np.empty((0,0)) # nx3 matrix of mesh points (x, y, z)
        self.X    = np.empty((0,0,0)) # 3D meshgrid for X coordinates
        self.Y    = np.empty((0,0,0)) # 3D meshgrid for Y coordinates
        self.Z    = np.empty((0,0,0)) # 3D meshgrid for Z coordinates
        self.snapshots = np.empty((0, 0, 0, 0)) # 4D array to store intermediate mesh states
        self._validate_model_params()
        
    def _validate_model_params(self):
        """Validate the model parameters."""
        # Check and accept resolution as a single value or a tuple of 3 values
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution, self.resolution)
        elif isinstance(self.resolution, tuple):
            assert len(self.resolution) == 3, "Resolution must be a single value or a tuple of 3 values."
        else:
            raise ValueError("Resolution must be a single value or a tuple of 3 values.")
        
        # Check and accept bounds as a single tuple of 2 values or a tuple of 3 tuples
        if isinstance(self.bounds[0], tuple):
            assert len(self.bounds) == 3, "Bounds must be a tuple of 3 tuples for x, y, and z dimensions."
        elif isinstance(self.bounds, tuple):
            assert len(self.bounds) == 2, "Bounds must be a tuple of 2 values for a single dimension."
            self.bounds = (self.bounds, self.bounds, self.bounds)
        else:
            raise ValueError("Bounds must be a tuple of 2 values or a tuple of 3 tuples.")
        
    def setup_mesh(self):
        """Sets up the 3D meshgrid based on given bounds and resolution."""
        # Unpack bounds and resolution
        try:
            x_bounds, y_bounds, z_bounds = self.bounds  
        except ValueError:
            print("Bounds must be a tuple of 3 tuples for x, y, and z dimensions.")
            print("Bounds: ", self.bounds)
            print(f"Length: {len(self.bounds)}")
            raise
        x_res, y_res, z_res = self.resolution

        # Create linspace for x, y, z
        x = np.linspace(*x_bounds, num=x_res, dtype=self.dtype)
        y = np.linspace(*y_bounds, num=y_res, dtype=self.dtype)
        z = np.linspace(*z_bounds, num=z_res, dtype=self.dtype)
        
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
        for event in history:
            if not isinstance(event, GeoProcess):
                msg = f"All items in the history list must be instances of the GeoProcess class. Found {type(event)}. for {event}"
                raise TypeError(msg)  
        
        # Extend the existing history with the new history
        self.history.extend(history)
        
    def get_history_string(self):
        """Returns a string describing the complete geological history of the model."""
        if not self.history:
            return "No geological history to display."
        
        history_str = "Geological History:\n"
        for index, process in enumerate(self.history):
            history_str += f"{index + 1}: {str(process)}\n"
        
        return history_str
            
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
        
        # Clear the model data before recomputing
        self.clear_data()
        # Allocate memory for the mesh and data
        self.setup_mesh()        
        # Determine how many snapshots are needed for memory pre-allocation
        self._prepare_snapshots()    
        # Backward pass to reverse mesh grid of points
        self._backward_pass()        
        # Forward pass to apply deposition events
        self._forward_pass()
        
        # TODO: Might be nice to have an option to visualize the model at different stages
        # Clean up snapshots taken during the backward pass
        self.snapshots = np.empty((0, 0, 0, 0))
          
    def _prepare_snapshots(self):
        """ Determine when to take snapshots of the mesh during the backward pass.
        
        Snapshots of the xyz mesh should be taken at end of a transformation sequence
        """  
        # Always include the time start state of mesh      
        snapshot_indices = [0]
        for i in range(1,len(self.history)):
            if isinstance(self.history[i], Deposition) and isinstance(self.history[i-1], Transformation):
                snapshot_indices.append(i)
        
        self.snapshot_indices = snapshot_indices
        
        self.snapshots = np.empty((len(self.snapshot_indices), *self.xyz.shape))
        log.info(f"Intermediate mesh states will be saved at {self.snapshot_indices}")
        log.info(f"Total gigabytes of memory required: {self.snapshots.nbytes * 1e-9:.2f}")
        
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
    
    def renormalize_height(self, new_max = 0, auto = False):
        """ Shift the model vertically so that the highest point in view field is at a new maximum height.
        Note this operation is expensive since it requires recomputing the model.
        
        Parameters:
        - new_max: The new maximum height for the model.
        - optional auto: Automatically select a new maximum height based on the model's current height.
        """
        assert self.data is not None, "Data array is empty."
        #Find the highest point
        valid_indices = ~np.isnan(self.data)
        valid_z_values = self.xyz[valid_indices, 2]
        try:
            current_max_z = np.max(valid_z_values)
        except ValueError:
            print("All data values are NaN, cannot renormalize.")
            zmin, zmax = self.get_z_bounds()
            current_max_z = zmin

        if auto:
            new_max = self.get_target_normalization()

        # Calculate the model shift required to shift to a desired maximum height
        shift_z = new_max - current_max_z
        
        # Add a shift transformation to the history and recompute
        self.add_history(Shift([0, 0, shift_z]))
        self.clear_data()
        self.compute_model()
    
    def get_target_normalization(self, target_max=.8, std_dev = 0.05):
        """ Get the normalization factor to scale the model to a target maximum height.
        
        Parameters:
        - target_max: The target maximum height for the model as a fraction of total height.
        - std_dev: The standard deviation of the normal distribution used to add variation.
        """
        bounds = self.get_z_bounds()
        zmin, zmax = bounds
        z_range = zmax - zmin
        target_height = zmin + z_range*np.random.normal(target_max, std_dev) 
        log.debug(f"Normalization Target Height: {target_height}")
        return target_height
        
    def get_z_bounds(self):
        """Return the minimum and maximum z-coordinates of the model."""
                # Check if bounds is a tuple of tuples (multi-dimensional)
        if isinstance(self.bounds[0], tuple):
            # Multi-dimensional bounds, assuming the last tuple represents the z-dimension
            z_vals = self.bounds[-1]
        else:
            # Single-dimensional bounds
            z_vals = self.bounds
        
        return z_vals
    
    def get_data_grid(self):
        """Return the model data."""
        return self.data.reshape(self.X.shape)   
    
    def add_topography(self, mesh):
        """Add a topography mesh to the model.
        
        Parameters:
        - mesh: A 2D numpy array representing the topography mesh.
        """
        
        # Ensure the mesh is 2D and matches the dimensions of the X and Y grids
        if mesh.shape != self.X.shape[:2]:
            raise ValueError("Topography mesh shape must match the X and Y dimensions of the model.")

        # Expand the 2D topography mesh to match the 3D volume
        expanded_mesh = np.repeat(mesh[:, :, np.newaxis], self.resolution[2], axis=2)
        
        # Set all z-values higher than the corresponding topo point at the xy column to np.nan
        above_topo_mask = self.Z > expanded_mesh
        
        # Reshape the data mesh points into a volume
        data = self.get_data_grid()
        # Add the topography mesh to the model by setting  
        
        data[above_topo_mask] = np.nan
        self.data = data.flatten()