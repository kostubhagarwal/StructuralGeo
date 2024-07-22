""" A collection of curated geological words that combine random variables and geoprocesses into a single generator class. """

import numpy as np
from typing import List
import structgeo.model as geo
import structgeo.probability as rv

class GeoWord():
    """ Base class providing structure for generating events"""
    
    def __init__(self):
        self.hist = []
        pass
    
    def build_history(self):
        raise NotImplementedError()
            
    def generate(self):
        """ Generate n_samples of the geological word. """
        self.hist.clear()
        # Generate random process        
        self.build_history()
        # Return the wrapped process
        return geo.CompoundProcess(self.hist)
    
class FineRepeatSediment(GeoWord):
    def build_history(self):
        # Finer Sediment
        sb = geo.SedimentBuilder(start_value=1, total_thickness=np.random.normal(800,100), min_layers=2, max_layers=5, std=0.5) 
        for _ in range(np.random.randint(1, 3)):
            sediment = geo.Sedimentation(*sb.build_layers())
            self.hist.append(sediment)
    
class CoarseRepeatSediment(GeoWord):
    # Generate a repeating sequence of coarse sediment layers            
    def build_history(self):   
        sb = geo.SedimentBuilder(start_value=1, total_thickness=np.random.normal(1000,100), min_layers=2, max_layers=5, std=0.5)   
        for _ in range(np.random.randint(1, 2)):
            sediment = geo.Sedimentation(*sb.build_layers())
            self.hist.append(sediment)
    
class MicroNoise(GeoWord):
    # Scatter shot small noise through model    
    def build_history(self):
        wave_generator = rv.FourierWaveGenerator(num_harmonics=np.random.randint(4, 6), smoothness=0.8)
        for _ in range(np.random.randint(3, 7)):
            fold_params = {
                'strike': np.random.uniform(0, 360),
                'dip': np.random.uniform(0, 360),
                'rake': np.random.uniform(0, 360),
                'period': np.random.uniform(200, 4000),
                'amplitude': np.random.uniform(5, 20),
                'periodic_func': wave_generator.generate()
            }
            fold = geo.Fold(**fold_params)
            self.hist.append(fold)
            
        
        

        
    
    


        
        