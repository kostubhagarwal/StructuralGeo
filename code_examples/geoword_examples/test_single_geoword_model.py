import time as clock

import pyvista as pv

import structgeo.model as geo
import structgeo.plot as geovis
from structgeo.generation import *

sed = EventSediment()

# List of geological words to generate
sentence = [EventBaseStrata(), EventSediment(), EventFold(), EventErosion(), EventSediment(), EventFold(), EventErosion()]
# Model resolution and boundse
z = 64
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
