import numpy as np
import time 
import logging
import matplotlib.pyplot as plt


import model.geo as geo
import model.plot as geovis
import model.history as history
import probability as rv

bedrock = geo.Bedrock(base=-5, value=0)
tilt = geo.Tilt(strike=0, dip=20)
upright_fold = geo.Fold(strike=0, dip=90, period = 20, amplitude = 2)
dike  = geo.Dike(strike=0, dip=60, width=3, point=[0, 0, 0], data_value=7)
upright_fold2 = geo.Fold(strike=110, dip=60, period = 40, amplitude = 2)

# Create a list of numbers from 0 to 4
sediment_rock_types = list(range(4,7))
# Shuffle the list in place
np.random.shuffle(sediment_rock_types)

sediment0 = geo.Sedimentation(height = 0, value_list= range(0,5),  
                              value_selector= rv.NonRepeatingRandomListSelector,
                              thickness_callable= lambda: np.random.lognormal(.5,.5)
                             )
sediment1 = geo.Sedimentation(height = 5, value_list= sediment_rock_types)

list_transformations = [bedrock, sediment0, tilt, sediment1, upright_fold, dike]
test_history1 = [sediment0, dike, tilt, sediment1, upright_fold, upright_fold2]


bounds = ((-20,20), (-20,20), (-10,10))
model = geo.GeoModel(bounds = bounds, resolution = 64)
model.add_history(test_history1)
model.compute_model()
model.fill_nans()

geovis.volmesh(model, threshold= -.5)

# fig, ax = geovis.volview(model)
# plt.show()
# geovis.plotCrossSection(model, coord='y', slice_index=32)

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

