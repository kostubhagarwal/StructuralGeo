import pyvista as pv

from structgeo.generation import *
import structgeo.model as geo
import structgeo.plot as geovis

import time as clock


# List of geological words to generate
sentence = [InfiniteSedimentMarkov()]
# Model resolution and bounds
z = 128
res = (2*z, 2*z, z)
bounds = (BOUNDS_X, BOUNDS_Y, BOUNDS_Z) # Bounds imported from generation (geowords file)

hist = generate_history(sentence)
# model = generate_normalized_model(hist, bounds, res)
model = geo.GeoModel(bounds=bounds, resolution=res)
model.add_history(hist)
model.add_history(geo.Sedimentation(value_list=[7], thickness_list=[2000]))
start = clock.time()
model.compute_model()
finish = clock.time()
print(f"Model computed in {finish-start:.2f} seconds.")

# geovis.transformationview(model).show()
print(model.get_history_string())