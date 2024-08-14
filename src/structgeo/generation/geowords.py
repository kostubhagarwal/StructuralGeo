""" A collection of curated geological words that combine random variables and geoprocesses into a single generator class. """

from typing import List, Union
import copy

import numpy as np

import structgeo.model as geo
import structgeo.probability as rv
from structgeo.probability import SedimentBuilder, MarkovSedimentHelper

BOUNDS_X = (-3840, 3840)
BOUNDS_Y = (-3840, 3840)
BOUNDS_Z = (-1920, 1920)
X_RANGE = BOUNDS_X[1] - BOUNDS_X[0]
Z_RANGE = BOUNDS_Z[1] - BOUNDS_Z[0]
BED_ROCK_VAL = 0
SEDIMENT_VALS = [1, 2, 3, 4, 5]
INTRUSION_VALS = [6, 7, 8, 9, 10]


class GeoWord:
    """
    Base class providing structure for generating events
    """

    def __init__(self, seed=None):
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

    def add_process(
        self,
        item: Union[geo.GeoProcess, "GeoWord", List[Union[geo.GeoProcess, "GeoWord"]]],
    ):
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
        self.add_process(geo.Bedrock(base=BOUNDS_Z[0], value=0))


class InfiniteSedimentUniform(GeoWord):  # Validated
    """A large sediment accumulation to simulate deep sedimentary layers."""

    def build_history(self):
        depth = (Z_RANGE) * (
            3 * geo.GeoModel.EXT_FACTOR
        )  # Pseudo-infinite using a large depth
        vals = []
        thicks = []
        while depth > 0:
            vals.append(self.rng.choice(SEDIMENT_VALS))
            thicks.append(self.rng.uniform(50, Z_RANGE / 4))
            depth -= thicks[-1]

        self.add_process(geo.Sedimentation(vals, thicks, base=depth))


class InfiniteSedimentMarkov(GeoWord):  # Validated
    """A large sediment accumulation to simulate deep sedimentary layers with dependency on previous layers."""

    def build_history(self):
        # Caution, the depth needs to extend beyond the bottom of the model mesh,
        # Including height bar extensions for height tracking, or it will leave a gap underneath
        depth = (Z_RANGE) * (3 * geo.GeoModel.EXT_FACTOR)

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(200, Z_RANGE / 4),
            thickness_variance=self.rng.uniform(0.1, 0.6),
            dirichlet_alpha=self.rng.uniform(0.6, 1.2),
            anticorrelation_factor=0.6,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        self.add_process(geo.Sedimentation(vals, thicks, base=BOUNDS_Z[0] - depth))


class InfiniteSedimentTilted(GeoWord):  # Validated
    """
    A large sediment accumulation to simulate deep sedimentary layers with a tilt.

    Fills entire model with sediment, then tilts the model, then truncates sediment to the bottom of the model.
    """

    def build_history(self):
        depth = (Z_RANGE) * (2 * geo.GeoModel.EXT_FACTOR + 1)

        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(150, Z_RANGE / 4),
            thickness_variance=self.rng.uniform(0.1, 0.6),
            dirichlet_alpha=self.rng.uniform(0.6, 2.0),
            anticorrelation_factor=0.3,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        # Sediment needs to go above and below model bounds for tilt, build from bottom up through extended model
        sed = geo.Sedimentation(
            vals, thicks, base=BOUNDS_Z[0] - Z_RANGE * geo.GeoModel.EXT_FACTOR
        )

        tilt = geo.Tilt(
            strike=self.rng.uniform(0, 360),
            dip=self.rng.normal(0, 10),
            origin=(0, 0, 0),
        )
        unc = geo.UnconformityBase(BOUNDS_Z[0])
        self.add_process([sed, tilt, unc])


""" Sediment Acumulation Events"""


class FineRepeatSediment(GeoWord):  # Validated
    """A series of thin sediment layers with repeating values."""

    def build_history(self):
        # Get a log-normal depth for the sediment block
        depth = self.rng.lognormal(
            *rv.log_normal_params(mean=Z_RANGE / 4, std_dev=Z_RANGE / 6)
        )

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(100, Z_RANGE / 10),
            thickness_variance=self.rng.uniform(0.1, 0.3),
            dirichlet_alpha=self.rng.uniform(
                0.03, 0.1
            ),  # Low alpha for high repeatability
            anticorrelation_factor=0.1,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        self.add_process(geo.Sedimentation(vals, thicks))


class CoarseRepeatSediment(GeoWord):  # Validated
    """A series of thick sediment layers with repeating values."""

    def build_history(self):
        # Get a log-normal depth for the sediment block
        depth = self.rng.lognormal(
            *rv.log_normal_params(mean=Z_RANGE / 3, std_dev=Z_RANGE / 8)
        )

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(Z_RANGE / 12, Z_RANGE / 6),
            thickness_variance=self.rng.uniform(0.1, 0.2),
            dirichlet_alpha=self.rng.uniform(
                0.03, 0.1
            ),  # Low alpha for high repeatability
            anticorrelation_factor=0.1,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        self.add_process(geo.Sedimentation(vals, thicks))


class SingleRandSediment(GeoWord):
    """A single sediment layer with a random value and thickness."""

    def build_history(self):
        val = self.rng.integers(1, 5)
        sediment = geo.Sedimentation(
            [val], [self.rng.normal(Z_RANGE / 5, Z_RANGE / 40)]
        )
        self.add_process(sediment)


""" Folding Events"""


class MicroNoise(GeoWord):  # Validated
    """A thin layer of noise to simulate small-scale sedimentary features."""

    def build_history(self):
        wave_generator = rv.FourierWaveGenerator(
            num_harmonics=self.rng.integers(4, 6), smoothness=0.8
        )

        for _ in range(self.rng.integers(3, 7)):
            period = self.rng.uniform(100, 1000)
            amplitude = period * self.rng.uniform(0.002, 0.005) + 5
            fold_params = {
                "origin": rv.random_point_in_ellipsoid((BOUNDS_X, BOUNDS_Y, BOUNDS_Z)),
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
        min_amp = period * 0.02
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
            "origin": rv.random_point_in_ellipsoid((BOUNDS_X, BOUNDS_Y, BOUNDS_Z)),
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
        min_amp = period * 0.01
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
            "origin": rv.random_point_in_ellipsoid((BOUNDS_X, BOUNDS_Y, BOUNDS_Z)),
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)


class FourierFold(GeoWord):  # Validated
    """A fold structure with a random number of harmonics."""

    def build_history(self):
        wave_generator = rv.FourierWaveGenerator(
            num_harmonics=np.random.randint(3, 7), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(500, 22000)
        min_amp = period * 0.02
        max_amp = period * (0.18 - 0.07 * period / 10000)  # Linear interp
        amp = self.rng.beta(a=1.2, b=1.5) * (max_amp - min_amp) + min_amp
        fold_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 45),
            "rake": self.rng.uniform(0, 360),
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)


""" Erosion events"""


class FlatUnconformity(GeoWord):
    """Flat unconformity down to a random depth"""

    MEAN_DEPTH = Z_RANGE / 8

    def build_history(self):
        factor = self.rng.lognormal(
            *rv.log_normal_params(mean=1, std_dev=0.5)
        )  # Generally between .25 and 2.5
        factor = np.clip(factor, 0.25, 3)
        depth = factor * self.MEAN_DEPTH
        print(f"Depth: {depth}")
        unconformity = geo.UnconformityDepth(depth)
        self.add_process(unconformity)


class TippedUnconformity(GeoWord):
    # TODO: Use tilt process instead of rotate
    def build_history(self):
        tilt_angle = np.random.normal(0, 20)
        theta = np.random.uniform(0, 360)
        x, y = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        rot_axis = np.array([x, y, 0])

        rot_in = geo.Rotate(rot_axis, tilt_angle)
        unconformity = geo.UnconformityDepth(np.random.uniform(0, 2000))
        rot_out = geo.Rotate(rot_axis, -tilt_angle)
        self.add_process([rot_in, unconformity, rot_out])


class WaveUnconformity(GeoWord):
    def build_history(self):
        fold_params = {
            "strike": np.random.uniform(0, 360),
            "dip": np.random.uniform(0, 360),
            "rake": np.random.uniform(0, 360),
            "period": np.random.uniform(100, 11000),
            "amplitude": np.random.uniform(200, 5000),
            "periodic_func": None,
        }
        fold_in = geo.Fold(**fold_params)
        fold_out = fold_in.copy()
        fold_out.amplitude *= -1 * np.random.normal(0.8, 0.1)
        unconformity = geo.UnconformityDepth(np.random.uniform(200, 2000))
        self.add_process([fold_in, unconformity, fold_out])


""" Intrusion Events"""


class SingleDikePlane(GeoWord):  # Validated
    def build_history(self):
        width = rv.beta_min_max(2, 4, 50, 500)
        dike_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 30),  # Bias towards vertical dikes
            "origin": rv.random_point_in_ellipsoid((BOUNDS_X, BOUNDS_Y, BOUNDS_Z)),
            "width": width,
            "value": self.rng.choice(INTRUSION_VALS),
        }
        dike = geo.DikePlane(**dike_params)
        self.add_process(dike)


class SingleDikeWarped(GeoWord):
    def build_history(self):
        strike = self.rng.uniform(0, 360)
        dip = self.rng.normal(90, 30)
        width = rv.beta_min_max(2, 4, 50, 500)
        dike_params = {
            "strike": strike,
            "dip": dip,  # Bias towards vertical dikes
            "origin": rv.random_point_in_ellipsoid((BOUNDS_X, BOUNDS_Y, BOUNDS_Z)),
            "width": width,
            "value": self.rng.choice(INTRUSION_VALS),
        }
        dike = geo.DikePlane(**dike_params)

        # Wrap the dike in a change of coordinates via fold
        fold_in = self.get_fold(strike, dip)
        fold_out = copy.deepcopy(fold_in)
        fold_out.amplitude *= -1

        self.add_process([fold_in, dike, fold_out])

    def get_fold(self, dike_strike, dike_dip):
        wave_generator = rv.FourierWaveGenerator(
            num_harmonics=np.random.randint(4, 9), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(1, 4) * X_RANGE
        min_amp = period * 0.02
        max_amp = period * 0.08
        amp = self.rng.beta(a=1.2, b=1.5) * (max_amp - min_amp) + min_amp
        amp = max_amp
        fold_params = {
            "strike": dike_strike + 90,
            "dip": (90 + dike_dip) / 2,  # average of dike dip and 90
            "rake": self.rng.normal(90, 10),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold
