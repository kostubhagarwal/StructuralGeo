import pyvista as pv

from structgeo.generation import *
import structgeo.model as geo
import structgeo.plot as geovis

import time as clock


# List of geological words to generate
sentence =  [InfiniteSedimentMarkov(), CoarseRepeatSediment(),SingleDikeWarped()]
# Model resolution and bounds
z = 128
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

geovis.transformationview(model).show()
print(model.get_history_string())
