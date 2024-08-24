import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import to_rgb
from pyvistaqt import BackgroundPlotter

import structgeo.model as geo
import structgeo.plot as geovis
import structgeo.probability as rv

pv.set_plot_theme("document")

bounds = (-3, 3)
resolution = 4
model = geo.GeoModel(bounds=bounds, resolution=resolution)

# Empty model grid representaiton
hist = []
# hist.append(geo.Bedrock(base=5000, value=1))#
rocktypes = list(range(1, 6))
rockthickness = [0.6 * 3] * 5
sediment = geo.Sedimentation(rocktypes, rockthickness)
hist.append(sediment)
model.add_history(hist)

model.clear_data()
model.compute_model()

# Create a plotter object
plotter = pv.Plotter(shape=(1, 3), window_size=[1800, 600], off_screen=True)
colormap = "viridis"

""" Sample Points Representation"""
plotter.subplot(0, 1)
# Point cloud representation
grid = pv.PolyData(model.xyz)
# Set data to the grid
values = model.data.flatten(order="F")  # Flatten the data in Fortran order
grid["values"] = values
plotter.add_mesh(
    grid,
    scalars=values,
    cmap=colormap,
    show_scalar_bar=False,
    style="points_gaussian",
    render_points_as_spheres=True,
    point_size=15,
)

# Show bounds
plotter.show_bounds(
    grid=True,
    location="outer",
    ticks="outside",
    n_xlabels=4,
    n_ylabels=4,
    n_zlabels=4,
    font_size=12,
    # show_xlabels=False, show_ylabels=False, show_zlabels=False,
    xtitle="Easting",
    ytitle="Northing",
    ztitle="Elevation",
    all_edges=True,
)
plotter.add_title("Discrete Samples", font_size=12, color="black")
# Set the view vector
plotter.view_vector([2, 2, 1])
# Increase the camera distance to zoom out
plotter.camera.zoom(0.8)

""" Voxels Representation"""
plotter.subplot(0, 2)
model.clear_data()
model.compute_model()

dimensions = tuple(x + 1 for x in model.resolution)
spacing = tuple((x[1] - x[0]) / (r - 1) for x, r in zip(model.bounds, model.resolution))
origin = tuple(x[0] - cs / 2 for x, cs in zip(model.bounds, spacing))

print(f"Dimensions: {dimensions}, Spacing: {spacing}")

grid = pv.ImageData(
    dimensions=dimensions,
    spacing=spacing,
    origin=origin,
)
# Necessary to reshape data vector in Fortran order to match the grid
grid["values"] = model.data.reshape(model.resolution).flatten(order="F")
plotter.add_mesh(
    grid,
    scalars="values",
    cmap=colormap,
    show_scalar_bar=False,
    show_edges=True,
)

# Show bounds
plotter.show_bounds(
    grid=True,
    location="outer",
    ticks="outside",
    font_size=12,
    # show_xlabels=False, show_ylabels=False, show_zlabels=False,
    xtitle="Easting",
    ytitle="Northing",
    ztitle="Elevation",
    all_edges=True,
)
plotter.add_title("Voxelized Samples", font_size=12, color="black")
plotter.view_vector([2, 2, 1])
# Increase the camera distance to zoom out
plotter.camera.zoom(0.9)

""" Continuous Representation"""
# comparison plot
plotter.subplot(0, 0)
# now rerender the same model but with high resolution
model.resolution = (128, 128, 128)
model.clear_data()
model.compute_model()

grid = pv.StructuredGrid(model.X, model.Y, model.Z)
values = model.data.reshape(model.X.shape)
grid["values"] = values.flatten(order="F")
mesh = grid

plotter.add_mesh(mesh, scalars="values", cmap=colormap, show_scalar_bar=False)
# Set the view vector
plotter.view_vector([2, 2, 1])
# Increase the camera distance to zoom out
plotter.camera.zoom(0.8)
# Add axes
_ = plotter.add_axes(line_width=6)
# Show bounds
plotter.show_bounds(
    grid="back",
    location="outer",
    ticks="outside",
    n_xlabels=3,
    n_ylabels=3,
    n_zlabels=3,
    font_size=12,
    # show_xlabels=False, show_ylabels=False, show_zlabels=False,
    xtitle="Easting",
    ytitle="Northing",
    ztitle="Elevation",
    all_edges=True,
)
plotter.add_title("Continuous Model", font_size=12, color="black")


plotter.show()
# save the plot to disk
plotter.screenshot("model_comparison.png")
