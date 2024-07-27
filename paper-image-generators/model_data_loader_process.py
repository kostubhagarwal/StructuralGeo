from structgeo.generation import *

import structgeo.plot as geovis
import pyvista as pv


yaml_loc = 'C:/Users/sghys/Summer2024/StructuralGeo/src/structgeo/generation/grammar_map.yml'    
def model_loader_test():
    loader = GeoModelGenerator(yaml_loc, model_resolution=(128,128,64)) 
    models = loader.generate_model_batch(16) 
    p = pv.Plotter(shape=(4, 4))
    for i, model in enumerate(models):
        p.subplot(i // 4, i % 4)
        geovis.volview(model, plotter=p)
    p.show()
     
model_loader_test()
    


