from structgeo.generation import *

import numpy as np
import structgeo.model as geo
import structgeo.plot as geovis
import structgeo.probability as rv
import pyvista as pv
import random

# preallocate an array to store generated histories
histories = []

# Setup a test grammar structure
grammar_structure = ['Sediment', 'Noise', 'Sediment', 'Noise']

# Generate 10 sentence samples
for _ in range(10):
    sentence = generate_sentence(vocabulary, grammar_structure)
    # generate 10 samples of each sentence
    h = generate_history(sentence, 10)
    histories.extend(h)
    
# Select a random set of 16 histories
selected_histories = random.sample(histories, 16)
p = pv.Plotter(shape=(4, 4))    
for i, hist in enumerate(selected_histories):
    p.subplot(i // 4, i % 4)
    model = geo.GeoModel(bounds=(-1920,1920),resolution=128,)
    model.add_history(hist)
    model.compute_model()
    geovis.volview(model, plotter=p)
    
p.show()
    


