import model as geo
import matplotlib.pyplot as plt
import numpy as np

layer0 = geo.Layer(base=-5., width=5., value=0)
layer1 = geo.Layer(base=0., width=5., value=1)
layer2 = geo.Layer(base=5., width=1., value=2)
layer3 = geo.Layer(base=6., width=2., value=3)

tilt = geo.Tilt(strike=0, dip=20)
upright_fold = geo.Fold(strike=0, dip=90, period = 40)
dike  = geo.Dike(strike=0, dip=60, width=3, point=[0, 0, 0], data_value=3)

model = geo.GeoModel(resolution = 50)

data = geo.ModelHistory(model.xyz, [layer0, layer1, layer2,  layer3, dike, tilt,upright_fold])

# Fillin the NaN's
indnan = np.isnan(data)
data[indnan] = 4

fig, ax = geo.volview(model.X, model.Y, model.Z, data)

plt.show()



print('Done')

