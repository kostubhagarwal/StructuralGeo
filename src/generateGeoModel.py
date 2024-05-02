import model.geo as geo
import model.plot as geovis
import matplotlib.pyplot as plt
import numpy as np

layer0 = geo.Layer(base=-5., width=5., value=0)
layer1 = geo.Layer(base=0., width=5., value=1)
layer2 = geo.Layer(base=5., width=1., value=2)
layer3 = geo.Layer(base=6., width=2., value=3)

tilt = geo.Tilt(strike=0, dip=20)
upright_fold = geo.Fold(strike=0, dip=90, period = 40)
dike  = geo.Dike(strike=0, dip=60, width=3, point=[0, 0, 0], data_value=3)

list_transformations = [layer0, layer1, layer2, layer3, dike, tilt, upright_fold]
# list_transformations = [layer0, layer1, layer2, layer3, dike, tilt]
model = geo.GeoModel(bounds = (-10,10), resolution = 64)

model.add_transformations(list_transformations)
model.compute_model()
nan_replacement_val = 4
model.fill_nans(nan_replacement_val)
fig, ax = geovis.volview(model)
plt.show()

geovis.plotCrossSection(model, coord='x', slice_index=0)






print('Done')

