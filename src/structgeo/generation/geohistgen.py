""" Collection of functions to generate histories and models using GeoWords and GeoProcesses. """

from typing import List, NamedTuple

import numpy as np

import structgeo.model as geo

from .geowords import GeoWord


def generate_sentence(vocabulary, grammar_structure):
    """Generate a sentence from a grammar structure."""
    sentence = [np.random.choice(vocabulary[word])() for word in grammar_structure]
    return sentence


def generate_history(sentence: List[GeoWord]) -> List[geo.GeoProcess]:
    """Generate a geological history from a sentence."""
    h = [word.generate() for word in sentence]
    return h