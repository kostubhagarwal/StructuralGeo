import numpy as np
import time 
import matplotlib.pyplot as plt

import model.geo as geo
import model.plot as geovis
import model.history as history
import probability as rv


# Generators
# Create a list of numbers from 0 to 4
sediment_rock_types = list(range(4,7))
# Shuffle the list in place
np.random.shuffle(sediment_rock_types)

# Depostions
bedrock = geo.Bedrock(base=-5, value=0)
dike  = geo.Dike(strike=45, dip=75, width=3, point=[0, 0, 0], data_value=7)

sediment0 = geo.Sedimentation(height = 0, value_list= range(1,5),  
                              value_selector= rv.NonRepeatingRandomListSelector,
                              thickness_callable= lambda: np.random.lognormal(.5,.5)
                             )
sediment1 = geo.Sedimentation(height = 5, value_list= sediment_rock_types)

# Transformations
tilt = geo.Tilt(strike=45, dip=20, origin=(-1,-1,0))
tilt2 = geo.Tilt(strike=10, dip=20, origin = (3,0,0))
upright_fold = geo.Fold(strike=65, dip=90, period = 20, amplitude = 1)
upright_fold2 = geo.Fold(strike=110, 
                         dip=30, 
                         period = 40, 
                         amplitude = 2, 
                         shape=1, 
                         origin=(0,0,0), 
                         periodic_func=rv.noisy_sine_wave(frequency=3, smoothing=10, noise_scale=0.1))
erosion_process = geo.ErosionLayer(thickness=2)

# Histories
test_history0 = [bedrock, sediment0, tilt, sediment1, dike, upright_fold,  upright_fold2]
test_history1 = [sediment0, sediment1, upright_fold2]
test_history2 = [bedrock, tilt2, sediment0]

bounds = ((-20,20), (-20,20), (-10,10))
model = geo.GeoModel(bounds = bounds, resolution = 256)
model.add_history(test_history0)
model.compute_model()
model.fill_nans()

# Three types of visualization
geovis.volview(model, threshold= -.5)
# geovis.orthsliceview(model, threshold= -.5)
#geovis.nsliceview(model, n=6, axis="x", threshold= -.5)




# config = {
#     'base_layers': {
#         'base_init': -5,
#         'mean_num': 4,    
#         'sigma_num': 1, 
#         'mean_width': 3,
#         'sigma_width': 1,
#     },
#     # Additional configuration parameters can be added as needed
# }

# sedimentary_history = history.SedimentaryHistory(config)

