""" Sentence structures and generation processes for geo histories. """

import numpy as np
from typing import List
import structgeo.model as geo
import structgeo.probability as rv
from .geowords import *

# Define the vocabulary of geological words in a dictionary that references classes from geowords
vocabulary = {
    'Sediment': [FineRepeatSediment, CoarseRepeatSediment],
    'Noise': [MicroNoise]
}


def generate_sentence(vocabulary, grammar_structure):
    """ Generate a sentence from a grammar structure. """
    sentence = []
    for word in grammar_structure:
        word_class = np.random.choice(vocabulary[word])
        sentence.append(word_class())    
    return sentence

def generate_history(sentence: List[GeoWord], n_samples: int) -> List[geo.GeoProcess]:
    """ Generate a list of geological histories from a sentence. """
    histories = []
    for _ in range(n_samples):
        h = [word.generate() for word in sentence]
        histories.append(h)
    return histories

    


