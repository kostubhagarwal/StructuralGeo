import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


def rotate(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def slip_normal_vectors(rake, dip, strike):
    # Calculate rotation matrices (Transform from fold to plane coordinates)
    M1 = rotate([0, 0, 1], -(rake))
    M2 = rotate([0.0, 1.0, 0], (dip))
    M3 = rotate([0, 0.0, 1.0], -(strike))

    # start with a north vector
    slip_coords = [0, 1.0, 0.0]
    # Rotate by the rake angle to get vector in fault plane coordinate
    fault_coords = M1 @ slip_coords
    # Fault coordinates plane is dipped along strike axis which is y-axis in fault plane
    strike_coords = M2 @ fault_coords
    # We need to rotate to reference north as y-axis
    slip_vector = M3 @ strike_coords

    # Trace the normal vector through same sequence of rotations
    U = M3 @ M2 @ M1 @ [0.0, 0.0, 1.0]
    U = U / np.linalg.norm(U)
    return slip_vector, U


def resample_mesh(mesh, resolution):
    """Resample a mesh to match a new x,y resolution.

    Parameters:
    mesh (np.ndarray): A 2D numpy array representing the mesh.
    resolution (tuple): A tuple of the new x and y resolution.
    """

    # Interpolate the topography mesh to match the model resolution
    mesh_x, mesh_y = mesh.shape
    resx, resy = resolution

    # Check for downsampling if a LPF is needed
    downsampling = resx < mesh_x or resy < mesh_y

    # TODO: Ask about the ideal sigma value when downsampling data
    if downsampling:
        # Apply Gaussian smoothing as a low-pass filter
        downsample_factor = (resx / mesh_x + resy / mesh_y) / 2
        # Popular choice online is to use sigma = 0.5 * downsample_factor
        sigma = 0.5 * downsample_factor
        mesh = gaussian_filter(mesh, sigma=sigma)

    # Define the original grid points, normalized between 0 and 1
    x = np.linspace(0, 1, mesh_x)
    y = np.linspace(0, 1, mesh_y)

    # Define the new grid points, normalized between 0 and 1
    model_x = np.linspace(0, 1, resx)
    model_y = np.linspace(0, 1, resy)
    X, Y = np.meshgrid(model_x, model_y, indexing="ij")

    # Sample the interpolalor at the given model mesh points
    interp = RegularGridInterpolator((x, y), mesh)
    model_mesh = interp((X, Y))

    return model_mesh
