import time as clock
import copy

import pyvista as pv
import numpy as np

import structgeo.model as geo
import structgeo.plot as geovis
from structgeo.generation import *

# func = FourierWaveGenerator(3, frequency=1/8500, smoothness=.5).generate()

# def thick_func(x, y):
#     return (2-((y/2000)**2))*(.5+.5*func(y)**2)

# strike = 0    
# dike1 = geo.DikePlane(strike=strike, dip=75, width=100, origin=(-6, 8, 0), value=9)
# disp_vec = np.array([np.cos(np.radians(strike)), np.sin(np.radians(strike)), 0])
# new_origin = dike1.origin + 500 * disp_vec
# print(f"Origin: {dike1.origin}, New origin: {new_origin}")
# dike2 = geo.DikePlane(strike=strike, dip=75, width=100, origin=new_origin, value=9)

# List of geological words to generate
sentence = [EventBaseStrata(), DikeGroup()]
# Model resolution and boundse
z = 32
res = (2 * z, 2 * z, z)
bounds = (
    BOUNDS_X,
    BOUNDS_Y,
    BOUNDS_Z,
)  # Bounds imported from generation (geowords file)

hist = generate_history(sentence)
start = clock.time()
model = generate_normalized_model(hist, bounds, res)
finish = clock.time()
print(f"Model computed in {finish-start:.2f} seconds.")

# geovis.transformationview(model).show()
geovis.categorical_grid_view(model).show()
print(model.get_history_string())
