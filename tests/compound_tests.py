import numpy as np
import pyvista as pv
import structgeo.model as geo
import structgeo.plot as geovis
import matplotlib.pyplot as plt

import pyvistaqt as pvqt

# # Define the function and its gradient
# x = np.linspace(0, 5, 100)
# y = -np.abs(x**3)
# vx = np.abs(3 * x**2) * np.sign(x)
# vy = np.ones_like(x)

# # Plot the function
# fig, ax = plt.subplots()
# ax.plot(x, y, label='Function')

# # Add quiver plot to show the gradient
# ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=10, color='r')
# ax.set_aspect('equal', 'box')

# # Add labels and legend
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.legend()

# # Pyvista vector field plot
# model = geo.GeoModel(bounds=(-10,10), resolution=32)
# model.add_history(geo.Sedimentation([1,2,3], [3]))
# model.compute_model()
# grid = pv.StructuredGrid(model.X, model.Y, model.Z)
# print(grid)
# print(grid.points)
# x = grid.points[:, 0]
# y = grid.points[:, 1]
# z = grid.points[:, 2]

# x = x/.5

# shape = 3
# # calculate normalized distance from z-axis
# r = np.sqrt(x**2 + y**2)/3
# z_surf = -np.abs(r**shape)
# # z distance calc
# dists = z - z_surf
# # take the gaussian of the distance
# dists = 4*np.exp(-.1*dists**2)


# vectors = np.vstack(
#     (
#         np.zeros_like(x),
#         np.zeros_like(y),
#         dists
#     ),
# ).T

# grid['vectors'] = vectors/4
# grid.set_active_vectors('vectors')

# p = pv.Plotter()
# p.add_mesh(grid.arrows, show_scalar_bar=False)
# p.add_bounding_box()
# p.show()


model = geo.GeoModel(bounds=(-1920,1920), resolution=128)
model.add_history(geo.Sedimentation([1,3,4,5], [500,200,300,200,500]))

# plug = geo.DikePlug(origin=(0,0,0),diam=100,minor_axis_scale=.4, rotation=30, shape=2.5, value=5, clip=False)
# model.add_history(plug)


# model.add_history(geo.PushHemisphere(origin=(0,0,-1000),diam=2000, height=350 , minor_axis_scale=.8, rotation=10, value=6, clip=False))
# model.add_history(geo.DikeHemisphere(origin=(0,0,-1000),diam=2000, height=350 , minor_axis_scale=.8, rotation=10, value=6, clip=False))
# model.add_history(geo.DikeColumn(origin=(0,0,-1000),diam=500, depth =-1600, minor_axis_scale=.3, rotation=60, value=6, clip=False))

model.add_history(geo.Laccolith(origin=(0,0,-2000),cap_diam=4000, stem_diam=300, height=600 , minor_axis_scale=.3, rotation=10, value=6))

model.compute_model()

p=geovis.categorical_grid_view(model)
p.show()