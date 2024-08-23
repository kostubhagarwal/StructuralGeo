import random

import numpy as np
import pyvista as pv

import structgeo.model as geo
import structgeo.plot as geovis
from structgeo.config import load_config
from structgeo.generation import *


# ---- Single Sentence Testing ---- #
def single_sentence_test():
    sentence = [
        InfiniteBasement(),
        FineRepeatSediment(),
        FineRepeatSediment(),
        MicroNoise(),
        NullWord(),
    ]
    histories = [generate_history(sentence) for _ in range(16)]
    # Select a random set of 16 histories
    selected_histories = random.sample(histories, 16)
    p = pv.Plotter(shape=(4, 4))
    for i, hist in enumerate(selected_histories):
        p.subplot(i // 4, i % 4)
        model = geo.GeoModel(bounds=(-1920, 1920), resolution=128)
        model.add_history(hist)
        model.compute_model(normalize=True)
        geovis.volview(model, plotter=p)
    p.show()


config = load_config(name="config_default.json")
yaml_loc = config["yaml_file"]


def model_loader_test():
    loader = YAMLGeostoryGenerator(config=yaml_loc, model_resolution=(128, 128, 64))
    models = loader.generate_models(16)
    p = pv.Plotter(shape=(4, 4))
    for i, model in enumerate(models):
        p.subplot(i // 4, i % 4)
        geovis.volview(model, plotter=p)
    p.show()

    print("")


# single_sentence_test()
model_loader_test()
