""" A collection of curated geological words that combine random variables and geoprocesses into a single generator class. """
from typing import List, Union

import numpy as np
from scipy.stats import lognorm

import structgeo.model as geo
import structgeo.probability as rv
from structgeo.probability import SedimentBuilder, MarkovSedimentHelper

BOUNDS_X = (-3840, 3840)
BOUNDS_Y = (-3840, 3840)
BOUNDS_Z = (-1920, 1920)
X_RANGE = BOUNDS_X[1] - BOUNDS_X[0]
Z_RANGE = BOUNDS_Z[1] - BOUNDS_Z[0]
BED_ROCK_VAL  = 0
SEDIMENT_VALS = [1,2,3,4,5]
INTRUSION_VALS = [6,7,8,9,10]
class GeoWord:
    """ 
    Base class providing structure for generating events 
    """
    
    def __init__(self, seed = None):
        self.hist = []
        self.rng = np.random.default_rng(seed)
    
    def build_history(self):
        """
        Builds a geological history for the word. Must be implemented by subclasses.
        """
        raise NotImplementedError()
            
    def generate(self):
        """ 
        Generate geological word assigning all rvs to geoprocesses
        """
        self.hist.clear()
        self.build_history()
        geoprocess = geo.CompoundProcess(self.hist.copy(), name=self.__class__.__name__)
        return geoprocess
    
    def add_process(self, item: Union[geo.GeoProcess, 'GeoWord', List[Union[geo.GeoProcess, 'GeoWord']]]):
        """
        Adds a GeoProcess, GeoWord or a list of processes/words to the GeoWord history. Used during the build_history method.
        Process and/or GeoWord objects should be added in the chronological order from earliest to latest event.

        Parameters
        ----------
        item : Union[GeoProcess, 'GeoWord', List[Union[GeoProcess, 'GeoWord']]]
            The item to add, which can be a GeoProcess, GeoWord, or a list containing either.

        Raises
        ------
        ValueError
            If the provided item is not a GeoWord, GeoProcess, or a list of these.
        """
        if isinstance(item, GeoWord):
            self.hist.extend(item.generate().history)
        elif isinstance(item, geo.GeoProcess):
            self.hist.append(item)
        elif isinstance(item, list):
            for sub_item in item:
                self.add_process(sub_item)
        else:
            raise ValueError(f"Expected GeoWord, GeoProcess, or list of these, got {type(item)}")

""" Identity word for generating a null event. """
class NullWord(GeoWord): # Validated
    """A null geological event, generating a process that does nothing."""
    def build_history(self):
        self.add_process(geo.NullProcess())

""" Infinite foundation layer(s) for initial model"""        
class InfiniteBasement(GeoWord): # Validated
    """A foundational bedrock layer to simulate an infinite basement."""
    def build_history(self):
        # Generate a simple basement layer
        self.add_process(geo.Bedrock(base=BOUNDS_Z[0], value=0))
        
class InfiniteSedimentUniform(GeoWord): # Validated
    """A large sediment accumulation to simulate deep sedimentary layers."""
    def build_history(self):
        depth = (Z_RANGE)*(3*geo.GeoModel.EXT_FACTOR) # Pseudo-infinite using a large depth
        vals =[]
        thicks = []
        while depth > 0:
            vals.append(self.rng.choice(SEDIMENT_VALS))
            thicks.append(self.rng.uniform(50,Z_RANGE/4))
            depth -= thicks[-1]
            
        self.add_process(geo.Sedimentation(vals, thicks, base=depth))
        
class InfiniteSedimentMarkov(GeoWord): # Validated
    """A large sediment accumulation to simulate deep sedimentary layers with dependency on previous layers."""
    
    def build_history(self):
        # Caution, the depth needs to extend beyond the bottom of the model mesh,
        # Including height bar extensions for height tracking, or it will leave a gap underneath
        depth = (Z_RANGE) * (3*geo.GeoModel.EXT_FACTOR)
        
        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(categories=SEDIMENT_VALS, 
                                             rng=self.rng, 
                                             thickness_bounds=(150, Z_RANGE/4),
                                             thickness_variance=self.rng.uniform(0.1,.6),
                                             dirichlet_alpha=self.rng.uniform(0.6, 2.0),
                                             anticorrelation_factor=.3
                                             )
        
        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        
        self.add_process(geo.Sedimentation(vals, thicks, base=BOUNDS_Z[0]-depth))
        
class InfiniteSedimentTilted(GeoWord): # Validated
    """
    A large sediment accumulation to simulate deep sedimentary layers with a tilt.
    
    Fills entire model with sediment, then tilts the model, then truncates sediment to the bottom of the model.
    """
    def build_history(self):
        depth = (Z_RANGE) * (2*geo.GeoModel.EXT_FACTOR + 1)
        
        markov_helper = MarkovSedimentHelper(categories=SEDIMENT_VALS, 
                                        rng=self.rng, 
                                        thickness_bounds=(150, Z_RANGE/4),
                                        thickness_variance=self.rng.uniform(0.1,.6),
                                        dirichlet_alpha=self.rng.uniform(0.6, 2.0),
                                        anticorrelation_factor=.3
                                        )
        
        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        #Sediment needs to go above and below model bounds for tilt, build from bottom up through extended model
        sed = geo.Sedimentation(vals, thicks, base=BOUNDS_Z[0] - Z_RANGE*geo.GeoModel.EXT_FACTOR)
        
        tilt = geo.Tilt(strike=self.rng.uniform(0, 360),
                        dip=self.rng.normal(0, 10),
                        origin=(0,0,0)
        )
        unc = geo.UnconformityBase(BOUNDS_Z[0])
        self.add_process([sed, tilt, unc])
            
""" Sediment Acumulation Events"""    
class FineRepeatSediment(GeoWord): # Validated
    """A series of thin sediment layers with repeating values."""
    def build_history(self):
        # Get a log-normal depth for the sediment block
        depth = self.rng.lognormal(*rv.log_normal_params(mean = Z_RANGE/4, std_dev = Z_RANGE/6)) 
        
        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(categories=SEDIMENT_VALS, 
                                             rng=self.rng, 
                                             thickness_bounds=(100, Z_RANGE/10),
                                             thickness_variance=self.rng.uniform(0.1,.3),
                                             dirichlet_alpha=self.rng.uniform(.03,.1), # Low alpha for high repeatability
                                             anticorrelation_factor=.1
                                             )
        
        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        
        self.add_process(geo.Sedimentation(vals, thicks))
    
class CoarseRepeatSediment(GeoWord): # Validated
    """A series of thick sediment layers with repeating values."""          
    def build_history(self):   
        # Get a log-normal depth for the sediment block
        depth = self.rng.lognormal(*rv.log_normal_params(mean = Z_RANGE/3, std_dev = Z_RANGE/8)) 
        
        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(categories=SEDIMENT_VALS, 
                                             rng=self.rng, 
                                             thickness_bounds=(Z_RANGE/12, Z_RANGE/6),
                                             thickness_variance=self.rng.uniform(0.1,.2),
                                             dirichlet_alpha=self.rng.uniform(.03,.1), # Low alpha for high repeatability
                                             anticorrelation_factor=.1
                                             )
        
        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        
        self.add_process(geo.Sedimentation(vals, thicks))
            
class SingleRandSediment(GeoWord):
    """A single sediment layer with a random value and thickness."""
    def build_history(self):
        val = self.rng.integers(1, 5)
        sediment = geo.Sedimentation([val], [self.rng.normal(Z_RANGE/5, Z_RANGE/40)])
        self.add_process(sediment)
             
class MicroNoise(GeoWord): # Validated
    """A thin layer of noise to simulate small-scale sedimentary features.""" 
    def build_history(self):
        wave_generator = rv.FourierWaveGenerator(num_harmonics=self.rng.integers(4, 6), smoothness=0.8)

        for _ in range( self.rng.integers(3, 7)):
            period = self.rng.uniform(100,1000)
            amplitude = period *  self.rng.uniform(0.002, 0.005) + 5
            fold_params = {
                'origin' : rv.random_point_in_ellipsoid((BOUNDS_X, BOUNDS_Y, BOUNDS_Z)),
                'strike':  self.rng.uniform(0, 360),
                'dip':  self.rng.uniform(0, 360),
                'rake':  self.rng.uniform(0, 360),
                'period': period,
                'amplitude': amplitude,
                'periodic_func': wave_generator.generate()
            }
            fold = geo.Fold(**fold_params)
            self.add_process(fold)
          
class SimpleFold(GeoWord):
    """A simple fold structure with random orientation and amplitude."""
    def build_history(self):
        period = np.random.uniform(100, 11000)
        proportion = np.random.normal(.05,.03)
        fold_params = {
            'strike': np.random.uniform(0, 360),
            'dip': np.random.uniform(0, 360),
            'rake': np.random.uniform(0, 360),
            'period': period,
            'amplitude': period*proportion,
            'periodic_func': None
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)

class ShapedFold(GeoWord):
    """ A fold structure with a random shape factor."""
    def build_history(self):
        period = np.random.uniform(100, 11000)
        proportion = np.random.normal(.05,.03)
        fold_params = {
            'strike': np.random.uniform(0, 360),
            'dip': np.random.uniform(0, 360),
            'rake': np.random.uniform(0, 360),
            'period': period,
            'amplitude': period*proportion,
            'shape':  np.random.normal(1,1),
            'periodic_func': None,
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)
        
class FourierFold(GeoWord):
    """ A fold structure with a random number of harmonics."""
    def build_history(self):
        wave_generator = rv.FourierWaveGenerator(num_harmonics=np.random.randint(4, 6), smoothness=np.random.normal(1,.2))
        period = np.random.uniform(500, 22000)
        proportion = np.random.normal(.05,.03)
        fold_params = {
            'strike': np.random.uniform(0, 360),
            'dip': np.random.uniform(0, 360),
            'rake': np.random.uniform(0, 360),
            'period': period,
            'amplitude': period*proportion,
            'periodic_func': wave_generator.generate()
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)

""" Erosion events"""        
class FlatUnconformity(GeoWord):
    def build_history(self):
        unconformity = geo.UnconformityDepth(np.random.uniform(0, 2000))
        self.add_process(unconformity)
        
class TippedUnconformity(GeoWord):
    #TODO: Use tilt process instead of rotate
    def build_history(self):
        tilt_angle = np.random.normal(0,20)
        theta = np.random.uniform(0, 360)
        x,y = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        rot_axis = np.array([x,y,0])
        
        rot_in = geo.Rotate(rot_axis, tilt_angle)        
        unconformity = geo.UnconformityDepth(np.random.uniform(0, 2000))
        rot_out = geo.Rotate(rot_axis, -tilt_angle)        
        self.add_process([rot_in, unconformity, rot_out])
        
class WaveUnconformity(GeoWord):
    def build_history(self):
        fold_params = {
            'strike': np.random.uniform(0, 360),
            'dip': np.random.uniform(0, 360),
            'rake': np.random.uniform(0, 360),
            'period': np.random.uniform(100, 11000),
            'amplitude': np.random.uniform(200, 5000),
            'periodic_func': None
        }
        fold_in = geo.Fold(**fold_params)
        fold_out = fold_in.copy()
        fold_out.amplitude *= -1*np.random.normal(.8,.1)
        unconformity = geo.UnconformityDepth(np.random.uniform(200, 2000))        
        self.add_process([fold_in, unconformity, fold_out])
            
        
        

        
    
    


        
        