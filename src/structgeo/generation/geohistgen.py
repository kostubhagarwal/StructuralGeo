""" Collection of functions to generate histories and models using GeoWords and GeoProcesses. """
import numpy as np
from typing import List, NamedTuple

import structgeo.model as geo
from .geowords import *

# TODO: fix model height normalization and cleanup general functions
def generate_sentence(vocabulary, grammar_structure):
    """ Generate a sentence from a grammar structure. """   
    sentence = [np.random.choice(vocabulary[word])() for word in grammar_structure]    
    return sentence

def generate_history(sentence: List[GeoWord]) -> List[geo.GeoProcess]:
    """ Generate a geological history from a sentence. """
    h = [word.generate() for word in sentence]
    return h

def generate_normalized_model(hist: List[geo.GeoProcess], bounds=(-1920,1920), resolution=128) -> geo.GeoModel:
    """ Generate a model from a history and normalize the height. Use a low resolution to estimate renormalization."""
    
    # Generate a low resolution model to estimate the renormalization
    model = geo.GeoModel(bounds=bounds, resolution=(8,8,64))
    model.add_history(hist)
    model.compute_model()
    
    # First squash the model downwards until it is below the top of the bounds
    new_max = model.get_target_normalization(target_max = .1)
    model_max = model.bounds[2][1]
    max_iter = 10
    while True and max_iter > 0:
        observed_max = model.renormalize_height(new_max=new_max)
        max_iter -= 1
        if observed_max < model_max:
            break
        
    # Now renormalize the model to the correct height
    model.renormalize_height(auto=True)
    # Copy the vertical shift required to normalize the model    
    normed_hist = model.history.copy()
    
    # Generate the final model with the correct resolution and normalized height
    model = geo.GeoModel(bounds=bounds, resolution=resolution)
    model.add_history(normed_hist)  
    model.clear_data()
    model.compute_model()
    return model