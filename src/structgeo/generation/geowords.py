""" A collection of curated geological words that combine random variables and geoprocesses into a single generator class. """

import copy
from typing import List, Union

import numpy as np

import structgeo.model as geo
import structgeo.probability as rv
from structgeo.probability import FourierWaveGenerator, MarkovSedimentHelper

BOUNDS_X = (-3840, 3840)
BOUNDS_Y = (-3840, 3840)
BOUNDS_Z = (-1920, 1920)
MAX_BOUNDS = (BOUNDS_X, BOUNDS_Y, BOUNDS_Z) # Intended maximum bounds for the model
X_RANGE = BOUNDS_X[1] - BOUNDS_X[0]
Z_RANGE = BOUNDS_Z[1] - BOUNDS_Z[0]
BED_ROCK_VAL = 0
SEDIMENT_VALS = [1, 2, 3, 4, 5]
DIKE_VALS = [6, 7]
INTRUSION_VALS = [8, 9]
BLOB_VALS = [10, 11]
# A target mean for random sedimentation depth
MEAN_SEDIMENTATION_DEPTH = Z_RANGE / 4

class GeoWord:
    """
    Base class for generating geological events within a hierarchical structure.

    The `GeoWord` class forms the foundation for constructing tree-like histories of geological processes. 
    Each instance represents a node in this structure, which can either branch into further `GeoWord` events 
    or terminate with one or more defined `GeoProcess` instances.
    """

    def __init__(self, seed: int =None):
        """
        Initializes a GeoWord instance with a history bucket and random number generator.

        Parameters
        ----------
        seed : Optional[int]
            An optional seed for the random number generator, ensuring reproducibility.
        """
        self.hist = []
        self.rng = np.random.default_rng(seed)

    def build_history(self):
        """
        Constructs the geological history for the GeoWord.

        This method should be overridden by subclasses to define the geological history for the event.
        History is built by adding GeoProcess, GeoWords, or lists of these using the `add_process` method.
        """
        raise NotImplementedError()

    def generate(self):
        """
        Generates the geological history by building and compiling it into a CompoundProcess.

        Returns
        -------
        geo.CompoundProcess
            A sampled geological history snippet with a CompoundProcess wrapper
        """
        self.hist.clear()
        self.build_history()
        geoprocess = geo.CompoundProcess(self.hist.copy(), name=self.__class__.__name__)
        return geoprocess

    def add_process(
        self,
        item: Union[geo.GeoProcess, "GeoWord", List[Union[geo.GeoProcess, "GeoWord"]]],
    ):
        """
        Adds a process or event to the GeoWord history.

        This method supports adding individual `GeoProcess` or `GeoWord` instances, as well as lists of them. 
        Items are added in chronological order from earliest to latest event.

        Parameters
        ----------
        item : Union[geo.GeoProcess, GeoWord, List[Union[geo.GeoProcess, GeoWord]]]
            The process or event to be added to the history.

        Raises
        ------
        ValueError
            If the item is not a `GeoProcess`, `GeoWord`, or a list of these.
        """
        if   isinstance(item, GeoWord):
            self.hist.extend(item.generate().history)
        elif isinstance(item, geo.GeoProcess):
            self.hist.append(item)
        elif isinstance(item, list):
            for sub_item in item:
                self.add_process(sub_item)
        else:
            raise ValueError(
                f"Expected GeoWord, GeoProcess, or list of these, got {type(item)}"
            )

""" Identity word for generating a null event. """


class NullWord(GeoWord):  # Validated
    """A null geological event, generating a process that does nothing."""

    def build_history(self):
        self.add_process(geo.NullProcess())


""" Infinite foundation layer(s) for initial model"""


class InfiniteBasement(GeoWord):  # Validated
    """A foundational bedrock layer to simulate an infinite basement."""

    def build_history(self):
        # Generate a simple basement layer
        self.add_process(geo.Bedrock(base=0, value=BED_ROCK_VAL))


class InfiniteSedimentUniform(GeoWord):  # Validated
    """A large sediment accumulation to simulate deep sedimentary layers."""

    def build_history(self):
        depth = (Z_RANGE) * (
            3 * geo.GeoModel.EXT_FACTOR
        )  # Pseudo-infinite using a large depth
        sediment_base = -depth
        
        vals = []
        thicks = []
        while depth > 0:
            vals.append(self.rng.choice(SEDIMENT_VALS))
            thicks.append(self.rng.uniform(50, Z_RANGE / 4))
            depth -= thicks[-1]
            
        self.add_process(geo.Bedrock(base=sediment_base, value=BED_ROCK_VAL))
        self.add_process(geo.Sedimentation(vals, thicks, base=sediment_base))


class InfiniteSedimentMarkov(GeoWord):  # Validated
    """A large sediment accumulation to simulate deep sedimentary layers with dependency on previous layers."""

    def build_history(self):
        # Caution, the depth needs to extend beyond the bottom of the model mesh,
        # Including height bar extensions for height tracking, or it will leave a gap underneath
        depth = (Z_RANGE) * (3 * geo.GeoModel.EXT_FACTOR)
        sediment_base = -depth

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(200, Z_RANGE / 4),
            thickness_variance=self.rng.uniform(0.1, 0.6),
            dirichlet_alpha=self.rng.uniform(0.6, 1.2),
            anticorrelation_factor=0.05,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        self.add_process(geo.Bedrock(base=sediment_base, value=BED_ROCK_VAL))
        self.add_process(geo.Sedimentation(vals, thicks, base=sediment_base))


class InfiniteSedimentTilted(GeoWord):  # Validated
    """
    A large sediment accumulation to simulate deep sedimentary layers with a tilt.

    Fills entire model with sediment, then tilts the model, then truncates sediment to the bottom of the model.
    """

    def build_history(self):
        depth = (Z_RANGE) * (6 * geo.GeoModel.EXT_FACTOR)
        sediment_base = -.5*depth

        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(200, Z_RANGE / 4),
            thickness_variance=self.rng.uniform(0.1, 0.6),
            dirichlet_alpha=self.rng.uniform(0.6, 1.2),
            anticorrelation_factor=0.05,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        # Sediment needs to go above and below model bounds for tilt, build from bottom up through extended model
        sed = geo.Sedimentation(
            vals, thicks, base=sediment_base
        )

        tilt = geo.Tilt(
            strike=self.rng.uniform(0, 360),
            dip=self.rng.normal(0, 10),
            origin=(0, 0, 0),
        )
        unc = geo.UnconformityBase(BOUNDS_Z[0])
        
        self.add_process(geo.Bedrock(base=sediment_base, value=BED_ROCK_VAL))
        self.add_process([sed, tilt, unc])

""" Sediment Acumulation Events"""

class FineRepeatSediment(GeoWord):  # Validated
    """A series of thin sediment layers with repeating values."""

    def build_history(self):
        # Get a log-normal depth for the sediment block
        depth = self.rng.lognormal(
            *rv.log_normal_params(mean=MEAN_SEDIMENTATION_DEPTH, std_dev=MEAN_SEDIMENTATION_DEPTH/3)
        )

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(100, Z_RANGE / 10),
            thickness_variance=self.rng.uniform(0.1, 0.3),
            dirichlet_alpha=self.rng.uniform(0.6, 2.0),  # Low alpha for high repeatability
            anticorrelation_factor=0.05,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        self.add_process(geo.Sedimentation(vals, thicks))


class CoarseRepeatSediment(GeoWord):  # Validated
    """A series of thick sediment layers with repeating values."""

    def build_history(self):
        # Get a log-normal depth for the sediment block
        depth = self.rng.lognormal(
            *rv.log_normal_params(mean=MEAN_SEDIMENTATION_DEPTH, std_dev=MEAN_SEDIMENTATION_DEPTH/3)
        )

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(Z_RANGE / 12, Z_RANGE / 6),
            thickness_variance=self.rng.uniform(0.1, 0.2),
            dirichlet_alpha=self.rng.uniform(
                .8,1.2
            ),  # Low alpha for high repeatability
            anticorrelation_factor=.05, # Low factor gives low repeatability
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        self.add_process(geo.Sedimentation(vals, thicks))


class SingleRandSediment(GeoWord):
    """A single sediment layer with a random value and thickness."""

    def build_history(self):
        val = self.rng.integers(1, 5)
        sediment = geo.Sedimentation(
            [val], [self.rng.normal(MEAN_SEDIMENTATION_DEPTH, MEAN_SEDIMENTATION_DEPTH/3)]
        )
        self.add_process(sediment)


""" Erosion events"""

class BaseErosionWord(GeoWord):  # Validated
    """Reusable generic class for calculating total depth of erosion events."""
    MEAN_DEPTH = Z_RANGE / 8
    
    def __init__(self, seed=None):
        super().__init__(seed)
        
    def calculate_depth(self):
        factor = self.rng.lognormal(
            *rv.log_normal_params(mean=1, std_dev=0.5)
        )  # Generally between .25 and 2.5
        factor = np.clip(factor, 0.25, 3)
        return factor * self.MEAN_DEPTH

class FlatUnconformity(BaseErosionWord): # Validated
    """Flat unconformity down to a random depth"""

    def build_history(self):
        total_depth = self.calculate_depth()
        unconformity = geo.UnconformityDepth(total_depth)
        self.add_process(unconformity)

    
class TiltedUnconformity(BaseErosionWord): # Validated
    """ Slightly tilted unconformity down to a random depth"""
    
    def build_history(self):
        num_tilts = self.rng.integers(1, 4)
        total_depth = self.calculate_depth()
        depths = np.random.dirichlet(alpha=[1]*num_tilts) * total_depth
        
        for depth in depths:
            strike = self.rng.uniform(0, 360)
            tilt_angle = self.rng.normal(0, 3)
            x,y,z = rv.random_point_in_ellipsoid(MAX_BOUNDS)
            origin = (x,y,0)
            tilt_in = geo.Tilt(strike=strike, dip=tilt_angle, origin=origin)
            tilt_out = geo.Tilt(strike=strike, dip=-tilt_angle, origin=origin)
            
            unconformity = geo.UnconformityDepth(depth)
            
            self.add_process([tilt_in, unconformity, tilt_out])

class WaveUnconformity(BaseErosionWord):
    """ Change of coordinates/basis with two orthogonal folds to create wavy unconformity"""
    
    def build_history(self):
        total_depth = self.calculate_depth()*.8
        orientation = self.rng.uniform(0, 360) # Principal orientation of the waves
        fold_in1, fold_out1 = self.get_fold_pair(strike = orientation, dip = 90)
        fold_in2, fold_out2 = self.get_fold_pair(strike = orientation + 90, dip = 90)
        unconformity = geo.UnconformityDepth(total_depth)
        self.add_process([fold_in1, fold_in2, unconformity, fold_out2, fold_out1])
        
    def get_fold_pair(self, strike, dip):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(3,5), smoothness=1  
        )
        period = self.rng.uniform(.5,2) * X_RANGE
        min_amp = period * 0.001
        max_amp = period * 0.03
        amp = rv.beta_min_max(a=1.5, b=1.5, min_val=min_amp, max_val=max_amp)
        fold_params = {
            "strike": strike,
            "dip": dip,  # average of dike dip and 90
            "rake": self.rng.normal(0, 10),  # Bias to vertical folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold_in = geo.Fold(**fold_params)
        fold_out = copy.deepcopy(fold_in)
        springback_factor = np.clip(self.rng.normal(.8, .1), a_min=0, a_max=1)
        fold_out.amplitude *= -1*springback_factor
        return fold_in, fold_out


""" Folding Events"""


class MicroNoise(GeoWord):  # Validated
    """A thin layer of noise to simulate small-scale sedimentary features."""

    def build_history(self):
        wave_generator = FourierWaveGenerator(
            num_harmonics=self.rng.integers(4, 6), smoothness=0.8
        )

        for _ in range(self.rng.integers(3, 7)):
            period = self.rng.uniform(100, 1000)
            amplitude = period * self.rng.uniform(0.002, 0.005) + 5
            fold_params = {
                "origin": rv.random_point_in_ellipsoid(MAX_BOUNDS),
                "strike": self.rng.uniform(0, 360),
                "dip": self.rng.uniform(0, 360),
                "rake": self.rng.uniform(0, 360),
                "period": period,
                "amplitude": amplitude,
                "periodic_func": wave_generator.generate(),
            }
            fold = geo.Fold(**fold_params)
            self.add_process(fold)


class SimpleFold(GeoWord):  # validated
    """A simple fold structure with random orientation and amplitude."""

    def build_history(self):
        period = rv.beta_min_max(a=1.4, b=1.4, min_val=100, max_val=14000)
        min_amp = period * 0.04
        max_amp = period * (
            0.18 - 0.07 * period / 10000
        )  # Linear interp, 1000 -> .17 , 11000 -> .10
        amp = self.rng.beta(a=2.1, b=1.4) * (max_amp - min_amp) + min_amp

        fold_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 45),  # Preference towards vertical fold planes
            "rake": self.rng.uniform(0, 360),
            "period": period,
            "amplitude": amp,
            "periodic_func": None,
            "phase": self.rng.uniform(0, 2 * np.pi),
            "origin": rv.random_point_in_ellipsoid(MAX_BOUNDS),
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)


class ShapedFold(GeoWord):  # Validated
    """A fold structure with a random shape factor."""

    def build_history(self):
        true_period = rv.beta_min_max(a=2.1, b=1.4, min_val=1000, max_val=11000)
        shape = self.rng.normal(0.3, 0.1)
        harmonic_weight = shape / np.sqrt(1 + shape**2)
        period = (
            1 - (2 / 3) * harmonic_weight
        ) * true_period  # Effective period due to shape
        min_amp = period * 0.04
        max_amp = period * (
            0.18 - 0.07 * period / 10000
        )  # Linear interp, 1000 -> .17 , 11000 -> .10
        amp = self.rng.beta(a=1.2, b=2.1) * (max_amp - min_amp) + min_amp

        fold_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 45),  # Preference towards vertical fold planes
            "rake": self.rng.uniform(0, 360),
            "period": true_period,
            "amplitude": amp,
            "shape": shape,
            "periodic_func": None,
            "phase": self.rng.uniform(0, 2 * np.pi),
            "origin": rv.random_point_in_ellipsoid(MAX_BOUNDS),
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)


class FourierFold(GeoWord):  # Validated
    """A fold structure with a random number of harmonics."""

    def build_history(self):

        period = self.rng.uniform(3000, 15000)
        mu_smoothness = 1.4 - 0.1 * period / 10000
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(3, 6),
            smoothness=np.random.normal(mu_smoothness, 0.2),
        )
        min_amp = period * 0.04
        max_amp = period * (0.18 - 0.09 * period / 10000)  # Linear interp
        amp = self.rng.beta(a=1.4, b=2.1) * (max_amp - min_amp) + min_amp
        fold_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 25),
            "rake": self.rng.uniform(0, 360),
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)


""" Dike Events"""


class DikePlaneWord(GeoWord):  # Validated
    def thickness_function(self, x, y):
        return np.exp(-np.abs(y) / 1000)
    
    def get_organic_thickness_func(self, length, wobble_factor=1.):
        # Make a fourier based modifier for both x and y
        fourier = rv.FourierWaveGenerator(num_harmonics=4, smoothness=1)
        x_var = fourier.generate()
        y_var = fourier.generate()
        amp = self.rng.uniform(0.1, 0.2)*wobble_factor # unevenness of the dike thickness        
        expo = self.rng.uniform(4,10) # Hyper ellipse exponent controls tapering sharpness

        def func(x, y):
            # Elliptical tapering thickness 0 at ends
            taper_factor = np.sqrt(np.maximum(1 - np.abs((2*y/length))**expo, 0))
            # The thickness modifier combines 2d fourier with tapering at ends
            return (1 + amp * x_var(x/Z_RANGE)) * (1 + amp * y_var(y/X_RANGE)) * taper_factor
        
        return func
    
    def build_history(self):
        width = rv.beta_min_max(2, 4, 50, 500)
        length = rv.beta_min_max(2, 2, 300, 16000)
        dike_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 10),  # Bias towards vertical dikes
            "origin": rv.random_point_in_ellipsoid(MAX_BOUNDS),
            "width": width,
            "value": self.rng.choice(INTRUSION_VALS),
            "thickness_func": self.get_organic_thickness_func(length, wobble_factor=np.uniform(.5,1.5)),
        }
        dike = geo.DikePlane(**dike_params)
        self.add_process(dike)


class SingleDikeWarped(DikePlaneWord):
    def build_history(self):
        strike = self.rng.uniform(0, 360)
        dip = self.rng.normal(90, 10)
        width = rv.beta_min_max(2, 4, 50, 500)
        length = rv.beta_min_max(2, 2, 300, 16000)
        dike_params = {
            "strike": strike,
            "dip": dip,  # Bias towards vertical dikes
            "origin": rv.random_point_in_ellipsoid(MAX_BOUNDS),
            "width": width,
            "value": self.rng.choice(INTRUSION_VALS),
            "thickness_func": self.get_organic_thickness_func(length, wobble_factor=1.5),
        }
        dike = geo.DikePlane(**dike_params)

        # Wrap the dike in a change of coordinates via fold
        fold_in = self.get_fold(strike, dip)
        fold_out = copy.deepcopy(fold_in)
        fold_out.amplitude *= -1

        self.add_process([fold_in, dike, fold_out])

    def get_fold(self, dike_strike, dike_dip):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(4, 8), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(.5, 2) * X_RANGE
        amp = self.rng.uniform(10,250)
        fold_params = {
            "strike": dike_strike + 90,
            "dip": (2*90 + dike_dip) / 3,  # weighted average of dike dip and 90
            "rake": self.rng.normal(90, 5),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold
    
class DikeGroup(DikePlaneWord):  
    def build_history(self):
        num_dikes = self.rng.integers(2, 6)              
        
        # Starting parameters, to be sequrntially modified
        origin = rv.random_point_in_ellipsoid(MAX_BOUNDS)
        strike = self.rng.uniform(0, 360)
        width = rv.beta_min_max(1.5, 4, 40, 400)  
        dip = self.rng.normal(90, 8)
        value = self.rng.choice(INTRUSION_VALS)
        spacing_avg = self.rng.lognormal(*rv.log_normal_params(mean=1500, std_dev=400))
        
        # Setup slight wave transform
        fold_in = self.get_fold(strike, dip)
        fold_out = copy.deepcopy(fold_in)
        fold_out.amplitude *= -1
        
        self.add_process(fold_in)
        
        for _ in range(num_dikes):
            length = rv.beta_min_max(2, 2, 600, 16000)
            dike_params = {
                "strike": strike,
                "dip": dip, 
                "origin": origin,
                "width": width,
                "value": value,
                "thickness_func": self.get_organic_thickness_func(length, wobble_factor=.5)
            }
            dike = geo.DikePlane(**dike_params)
            self.add_process(dike)
            
            # Modify parameters for next dike
            origin = self.get_next_origin(origin, strike, spacing_avg)
            strike += self.rng.normal(0, 2)
            width *= np.maximum(self.rng.normal(1, 0.1), .8)
            dip += self.rng.normal(0, 1)
            
        # Add final fold out
        self.add_process(fold_out)
    
    def get_next_origin(self, origin, strike, spacing_avg):
        # Move orthogonally to strike direction (strike measured from y-axis)
        orth_vec = np.array([np.cos(np.radians(strike)), -np.sin(np.radians(strike)), 0])
        orth_distance = spacing_avg * self.rng.uniform(0.9,1.3)
        # Shift a bit parallel to strike as well
        par_vec = np.array([-orth_vec[1], orth_vec[0], 0])
        par_distance =  spacing_avg * self.rng.uniform(-0.2,0.2)

        new_origin = origin + orth_distance * orth_vec + par_distance * par_vec
        return new_origin   

    def get_fold(self, dike_strike, dike_dip):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(4, 8), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(.5, 2) * X_RANGE
        amp = self.rng.uniform(30,60)
        fold_params = {
            "strike": dike_strike + 90,
            "dip": (2*90 + dike_dip) / 3,  # weighted average of dike dip and 90
            "rake": self.rng.normal(90, 5),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold