import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from pyvistaqt import BackgroundPlotter

import structgeo.model as geo
import structgeo.plot as geovis
import structgeo.probability as rv

import pickle as pkl

# Load geomodel from pickle file

model = geo.GeoModel
with open('C:/Users/sghys/Summer2024/StructuralGeo/paper-image-generators/model_scale30_ar1_res128.pkl', 'rb') as f:
    model = pkl.load(f)

model.history[14].strike = 30  
print(model.get_history_string())
model.compute_model()

mesh = geovis.get_mesh_from_model(model)

""" Plot the model with the snapshots of the transformation history."""
# Create the main plotter with a specified size and two viewports
p = pv.Plotter(shape=(1, 2), border=True, window_size=(1200, 600), off_screen=False)

p.subplot(0, 0)  # Left subplot (3/4 width)

time_arrow = pv.Arrow(start=(25000,0,4000), direction=(-1, 0, 0), scale=25000, tip_length=.20, tip_radius=0.01, shaft_radius=0.003,)
p.add_mesh(time_arrow, color='black', label='Time Arrow')

# Get final present-day mesh of model
final_mesh = geovis.get_mesh_from_model(model)  
plot_config = geovis.get_plot_config()    
if np.all(np.isnan(model.data)):
    p.add_text("No data to show, all values are NaN.", font_size=20)
else:
    # Add the mesh to the plotter
    p.add_mesh(final_mesh, scalars="values", 
                    **plot_config,
                    )
_ = p.add_axes(line_width=5)

resolution = model.resolution
    # Calculate the offset to separate each snapshot
# The offset is chosen based on the overall size of the model
x_offset = model.bounds[0][1] - model.bounds[0][0]  # Width of the model along x
x_offset = x_offset*1.15

# Remove first data time entry which is empty, add the final data time entry
data_snapshots = np.concatenate((model.data_snapshots[1:], model.data.reshape(1, -1)), axis=0)

# Reverse the snapshots
mesh_snapshots = model.mesh_snapshots[::-1]
data_snapshots = data_snapshots[::-1]

actors = []    
for i, (mesh_snapshot, data_snapshot) in enumerate(zip(mesh_snapshots, data_snapshots)):
    # Assuming snapshots are stored as Nx3 arrays
    # Reshape to 3D grid of points-- i.e. 4x4x4 grid of (x,y,z) points
    deformed_points = mesh_snapshot.reshape(resolution + (3,))
    grid = pv.StructuredGrid(deformed_points[..., 0] + (i+1) * x_offset * 1.3,  # Shift along x
                            deformed_points[..., 1], 
                            deformed_points[..., 2])
    # Set the same values to the new grid
    grid["values"] = data_snapshot.reshape(model.X.shape).flatten(order="F")  # Assigning scalar values to the grid       
    # Add grid to plotter with a unique color and using the same scalar values
    a = p.add_mesh(grid, style='wireframe', scalars="values", cmap = plot_config['cmap'], line_width=1, show_scalar_bar=False)
    actors.append(a)

# Create a plotter object
p.subplot(0, 1)  # Right subplot (1/4 width)
plot_config = geovis.get_plot_config()

if np.all(np.isnan(model.data)):
    p.add_text("No data to show, all values are NaN.", font_size=20)
else:
    # Add the mesh to the plotter
    p.add_mesh(mesh, scalars="values", 
                    **plot_config,                        
                    interpolate_before_map=False,
                    )
_ = p.add_axes(line_width=5)

# add a bounding box
flat_bounds = [item for sublist in model.bounds for item in sublist]
bounding_box = pv.Box(flat_bounds)
p.add_mesh(bounding_box, color="black", style="wireframe", line_width=1)


# Adjust subplot sizes
p.subplot(0, 0)
p.view_vector( [.2,1,.2] )   
p.camera.roll = 224
p.add_title("Depositions on a Deformed Mesh", font_size=12, color="black")

p.subplot(0, 1)
p.camera_position = 'iso'
p.camera.zoom(.7)
p.add_title("Final Model", font_size=12, color="black")

p.show()
p.screenshot("complex-transforms.png")