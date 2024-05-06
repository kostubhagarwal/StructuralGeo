import numpy as np

def rotate(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    
def slip_normal_vectors(rake, dip, strike):
    # Calculate rotation matrices (Transform from fold to plane coordinates)
    M1 = rotate([0, 0, 1], -(rake))
    M2 = rotate([1, 0, 0], -(dip))
    M3 = rotate([0, 0, 1], -(strike))

    # Start in slip vector coordinate frame as [1, 0, 0]
    slip_vector = [1.0, 0.0, 0.0]
    # Move to fault plane coordinates
    fault_coords = M1 @ slip_vector
    # Fault coordinates plane is dipped along strike axis which is x-axis in fault plane
    strike_coords = M2 @ fault_coords
    # We need to rotate to reference north as x-axis
    slip_vector = M3 @ strike_coords
    
    # Trace the normal vector through same sequence of rotations
    U = M3 @ M2 @ M1 @ [0., 0.0, 1.0]
    U= U / np.linalg.norm(U)
    return slip_vector, U