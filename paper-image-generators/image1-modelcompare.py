import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from pyvistaqt import BackgroundPlotter

import structgeo.model as geo
import structgeo.plot as geovis
import structgeo.probability as rv


def volview(model, threshold=-0.5, show_bounds = False) -> pv.Plotter:
    mesh = geovis.get_mesh_from_model(model, threshold)
    
    # Create a plotter object
    plotter = pv.Plotter()       # type: ignore
    plot_config = get_plot_config()
    
    if np.all(np.isnan(model.data)):
        plotter.add_text("No data to show, all values are NaN.", font_size=20)
    else:
        # Add the mesh to the plotter
        plotter.add_mesh(mesh, scalars="values", 
                        **plot_config,   
                        style='points',
                        point_size=15,                     
                        interpolate_before_map=False
                        )
    _ = plotter.add_axes(line_width=5)
    if show_bounds:
        plotter.show_bounds(
            grid='back',
            location='outer',
            ticks='outside',
            n_xlabels=3,
            n_ylabels=3,
            n_zlabels=3,
            xtitle='Easting',
            ytitle='Northing',
            ztitle='Elevation',
            all_edges=True,
        )
        
    # add a bounding box
    flat_bounds = [item for sublist in model.bounds for item in sublist]
    bounding_box = pv.Box(flat_bounds)
    plotter.add_mesh(bounding_box, color="black", style="wireframe", line_width=1)
    
    return plotter   

def transformationview(model, threshold=None):
    """ Plot the model with the snapshots of the transformation history."""
        
    # Create the plotter
    plotter = pv.Plotter()

    # Get final present-day mesh of model
    final_mesh = geovis.get_mesh_from_model(model, threshold)  
    plot_config = get_plot_config()    
    if np.all(np.isnan(model.data)):
        plotter.add_text("No data to show, all values are NaN.", font_size=20)
    else:
        # Add the mesh to the plotter
        plotter.add_mesh(final_mesh, scalars="values", 
                        **plot_config,
                        point_size=15,
                           
                        interpolate_before_map=False
                        )
    _ = plotter.add_axes(line_width=5)
    
    add_snapshots_to_plotter(plotter, model, plot_config['cmap'])
    return plotter

def add_snapshots_to_plotter(plotter, model, cmap):
    resolution = model.resolution
     # Calculate the offset to separate each snapshot
    # The offset is chosen based on the overall size of the model
    x_offset = model.bounds[0][1] - model.bounds[0][0]  # Width of the model along x
    
    # Remove first data time entry which is empty, add the final data time entry
    data_snapshots = np.concatenate((model.data_snapshots[1:], model.data.reshape(1, -1)), axis=0)
    
    # Reverse the snapshots
    mesh_snapshots = model.mesh_snapshots[::-1]
    data_snapshots = data_snapshots[::-1]
    
    actors = []    
    for i, (mesh_snapshot, data_snapshot) in enumerate(zip(mesh_snapshots, data_snapshots)):
        # Assuming snapshots are stored as Nx3 arrays
        deformed_points = mesh_snapshot.reshape(resolution + (3,))
        grid = pv.StructuredGrid(deformed_points[..., 0] + (i+1) * x_offset * 1.3,  # Shift along x
                                deformed_points[..., 1], 
                                deformed_points[..., 2])
        # Set the same values to the new grid
        grid["values"] = data_snapshot.reshape(model.X.shape).flatten(order="F")  # Assigning scalar values to the grid       
        # Add grid to plotter with a unique color and using the same scalar values
        a = plotter.add_mesh(grid, style='wireframe', scalars="values", cmap = cmap, line_width=1, show_scalar_bar=False)
        actors.append(a)
    
    return actors

bounds = (-1,1)
resolution = 4
model = geo.GeoModel(bounds=bounds, resolution=resolution)

# Empty model grid representaiton

hist = []
# hist.append(geo.Bedrock(base=5000, value=1))# 
rocktypes = list(range(1, 6))
rockthickness = [.6] * 5
sediment = geo.Sedimentation(rocktypes, rockthickness)
hist.append(sediment)
model.add_history(hist)

model.clear_data()
model.compute_model()


# Create a plotter object
plotter = pv.Plotter(shape=(1,2), window_size=[1200, 600], off_screen=True)       # type: ignore

""" Sample Points Representation"""
plotter.subplot(0,1)
# Point cloud representation
grid = pv.PolyData(model.xyz)
# Set data to the grid
values = model.data.flatten(order="F")  # Flatten the data in Fortran order
grid["values"] = values
colormap = 'viridis'
plotter.add_mesh(grid, scalars=values, cmap=colormap, show_scalar_bar=False,
                 point_size=25)
# Add axes
_ = plotter.add_axes(line_width=6)

# Show bounds
plotter.show_bounds(
    grid=True, location='outer', ticks='outside',
    n_xlabels=4, n_ylabels=4, n_zlabels=4,
    font_size=12,
    # show_xlabels=False, show_ylabels=False, show_zlabels=False,
    xtitle='Easting', ytitle='Northing', ztitle='Elevation',
    all_edges=True,
)
plotter.add_title("Discrete Samples", font_size=12, color="black")
# Set the view vector
plotter.view_vector([2, 2, 1])
# Increase the camera distance to zoom out
plotter.camera.zoom(0.8)

plotter.subplot(0,1)

""" Voxels Representation"""
# plotter.subplot(0,2)

# # Convert GeoModel data to ImageData
# spacing = ((model.bounds[0][1] - model.bounds[0][0]) / model.resolution[0],
#               (model.bounds[1][1] - model.bounds[1][0]) / model.resolution[1],
#                 (model.bounds[2][1] - model.bounds[2][0]) / model.resolution[2])
# origin = (model.bounds[0][0], model.bounds[1][0], model.bounds[2][0])
# values = model.data.reshape(model.X.shape)

# print(spacing, origin, values.shape)

# image_data = pv.ImageData(
#     dimensions=model.resolution,
#     spacing=spacing,
#     origin=origin,
# )

# # Assign the voxel values from your GeoModel to the ImageData
# image_data.point_data["values"] = model.data

# plotter.add_volume(image_data, cmap=colormap, show_scalar_bar=False)

""" Continuous Representation"""
# comparison plot
plotter.subplot(0,0)
# now rerender the same model but with high resolution
model.resolution = (128,128,128)
model.clear_data()
model.compute_model()

mesh = geovis.get_mesh_from_model(model)
plotter.add_mesh(mesh, scalars="values", cmap=colormap, show_scalar_bar = False)
# Set the view vector
plotter.view_vector([2, 2, 1])
# Increase the camera distance to zoom out
plotter.camera.zoom(0.8)
# Add axes
_ = plotter.add_axes(line_width=6)
# Show bounds
plotter.show_bounds(
    grid='back', location='outer', ticks='outside',
    n_xlabels=3, n_ylabels=3, n_zlabels=3,
    font_size=12,
    # show_xlabels=False, show_ylabels=False, show_zlabels=False,
    xtitle='Easting', ytitle='Northing', ztitle='Elevation',
    all_edges=True,
)
plotter.add_title("Continuous Model", font_size=12, color="black")


plotter.show()
# save the plot to disk
plotter.screenshot("model_comparison.png")
