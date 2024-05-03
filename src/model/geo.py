import numpy as np
from .util import rotate

class GeoModel:
    EMPTY_VALUE = -1
    MAX_VALUE = 10
    
    def __init__(self,  bounds=(0, 16), resolution=64):
        self.resolution = resolution
        self.bounds = bounds
        self.history = []
        self.snapshots = None
        # Init 3D meshgrid for X, Y, and Z coordinates of the view field
        x = np.linspace(*self.bounds, num=self.resolution)
        y = np.linspace(*self.bounds, num=self.resolution)
        z = np.linspace(*self.bounds, num=self.resolution)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        # Combine flattened arrays into a 2D numpy array where each row is an (x, y, z) coordinate
        # This gives mesh representation of the model
        self.xyz = np.column_stack((self.X.flatten(), self.Y.flatten(), self.Z.flatten()))
        # Initialize data array with NaNs
        self.data = np.full(self.xyz.shape[0], np.nan)  
        
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
        
        Backward pass:
        The xyz mesh is backtracked through history using the transformations
        stored in the history list. Intermediate states are stored at required
        intervals for the forward pass.
        
        Forward pass:
        The deposition events are applied to the xyz mesh in its intermediate 
        transformed state.
        
        """
        
        # Clone the xyz array to avoid modifying the original input
        xyz_motion = self.xyz.copy()

        for transformation in reversed(self.history):
            xyz_motion, self.data = transformation.run(xyz_motion, self.data)
 
        return self.data
    
    # def push_snapshot(self, xyz):
    #     """Push the current mesh state onto the snapshot stack."""
    #     self.snapshots.append(np.copy(self.xyz))

    # def pop_snapshot(self):
    #     """Pop the top state from the snapshot stack and restore it."""
    #     if self.snapshots:
    #         xyz = self.snapshots.pop()
    #     else:
    #         raise IndexError("Attempted to pop from an empty snapshot stack.")
    #     return xyz
    
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

        # Update data based on whether points are within the width of the dike
        data[np.abs(dists) <= self.width / 2.0] = self.data_value

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