from structgeo.generation import *

import numpy as np
import structgeo.model as geo
import structgeo.plot as geovis
import structgeo.probability as rv
import pyvista as pv
import random

# # preallocate an array to store generated histories
# histories = []

# # Setup a test grammar structure
# grammar_structure = ['Sediment', 'Noise', 'Sediment', 'Noise']

# # Generate 10 sentence samples
# for _ in range(10):
#     sentence = generate_sentence(vocabulary, grammar_structure)
#     # generate 10 samples of each sentence
#     h = generate_history(sentence, 10)
#     histories.extend(h)

# ---- Single Sentence Testing ---- #    
def single_sentence_test():
    sentence = [InfiniteBasement(), FineRepeatSediment(), FineRepeatSediment(), MicroNoise() , NullWord()]
    histories = [generate_history(sentence) for _ in range(16)]  
    # Select a random set of 16 histories
    selected_histories = random.sample(histories, 16)
    p = pv.Plotter(shape=(4, 4))    
    for i, hist in enumerate(selected_histories):
        p.subplot(i // 4, i % 4)
        model = generate_normalized_model(hist)
        geovis.volview(model, plotter=p)    
    p.show()

yaml_loc = 'C:/Users/sghys/Summer2024/StructuralGeo/src/structgeo/generation/grammar_map.yml'    
def model_loader_test():
    loader = GeologicalModelLoader(yaml_loc, model_resolution=(128,128,64)) 
    models = loader.generate_model_batch(16) 
    p = pv.Plotter(shape=(4, 4))
    for i, model in enumerate(models):
        p.subplot(i // 4, i % 4)
        geovis.volview(model, plotter=p)
    p.show()
   

    print('')
    
    
model_loader_test()
    


