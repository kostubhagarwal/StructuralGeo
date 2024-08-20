import time as clock
import copy

import pyvista as pv
import numpy as np

import structgeo.model as geo
import structgeo.plot as geovis
from structgeo.generation import *



def single_plotter():
    # List of geological words to generate
    sentence = [BaseStrata(),Laccolith()]
    # Model resolution and boundse
    z = 64
    res = (2 * z, 2 * z, z)
    bounds = (
        BOUNDS_X,
        BOUNDS_Y,
        BOUNDS_Z,
    )  # Bounds imported from generation (geowords file)e

    hist = generate_history(sentence)
    start = clock.time()
    model = geo.GeoModel(bounds=bounds, resolution=res)
    model.add_history(hist)
    model.compute_model(normalize=True)
    finish = clock.time()
    print(f"Model computed in {finish-start:.2f} seconds.")

    # geovis.transformationview(model).show()e
    geovis.categorical_grid_view(model).show()
    print(model.get_history_string())
    
    
def process_plotter():
    bed = geo.Bedrock(0,1)
    sediment_word = Sediment()
    sediment = sediment_word.generate()
    
    hemi = geo.DikeHemispherePushed(
        origin=(0,0,1500),
        diam  =2000,
        height=300,
        minor_axis_scale=.5,
        rotation=0,
        value=5,
        upper=True,
        clip=False,
        z_function=get_hemi_function(wobble_factor=.05),
    )
    
    hist = [bed, sediment_word.generate(), sediment_word.generate(), hemi]
    # Model resolution and boundse
    z = 64
    res = (2 * z, 2 * z, z)
    bounds = (
        BOUNDS_X,
        BOUNDS_Y,
        BOUNDS_Z,
    )  # Bounds imported from generation (geowords file)
    model = geo.GeoModel(bounds=bounds, resolution=res)
    model.add_history(hist)
    model.compute_model(normalize=True)
    geovis.categorical_grid_view(model).show()

if __name__ == "__main__":
    single_plotter()
    
