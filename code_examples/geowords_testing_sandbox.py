import random

import pyvista as pv

from structgeo.generation import *
import structgeo.model as geo
import structgeo.plot as geovis


def main():
    # List of geological words to generate
    sentence = [InfiniteBasement(), InfiniteSediment()]
    # Model resolution and bounds
    res = (64,64,32)
    bounds = ((-3840,3840),(-3840,3840),(-1920,1920)) 
    generate_samples(sentence, bounds, res)
    
    res = (64,64,64)
    bounds = ((-1920,1920),(-1920,1920),(-1920,1920)) 
    generate_samples(sentence, bounds, res)
    
def generate_samples(sentence, bounds, res):
    histories = [generate_history(sentence) for _ in range(16)]  
    p = pv.Plotter(shape=(4, 4))    
    for i, hist in enumerate(histories):
        p.subplot(i // 4, i % 4)
        model = generate_normalized_model(hist, bounds, res)
        geovis.volview(model, plotter=p)    
    p.show()
    
if __name__ == "__main__":
    main()    
    print('')