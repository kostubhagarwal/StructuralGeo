import time

import numpy as np

import structgeo.generation as gen
import structgeo.model as geo
import structgeo.plot as geovis

sentence = [
    gen.InfiniteBasement(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
]

# Save directory for models
DEFAULT_BASE_DIR = "../saved_models"

# Model resolution and bounds
res = (128, 128, 64)
bounds = ((-3840, 3840), (-3840, 3840), (-1920, 1920))
hist = gen.generate_history(sentence)

start = time.time()
# Generate a low resolution model to estimate the renormalization
model = geo.GeoModel(bounds=bounds, resolution=32, height_tracking=True)
model.add_history(hist)
model.compute_model()

# First squash the model downwards until it is below the top of the bounds
new_max = model.get_target_normalization(target_max=0.1)
model_max = model.bounds[2][1]
max_iter = 10
while True and max_iter > 0:
    observed_max = model.renormalize_height(new_max=new_max, recompute=True)
    max_iter -= 1
    if observed_max < model_max:
        break

# geovis.volview(model, show_bounds=True).show()

# Now renormalize the model to the correct height
model.renormalize_height(auto=True)
# Copy the vertical shift required to normalize the model
normed_hist = model.history.copy()
stop = time.time()
print(f"Normalization time: {stop-start}")

# Generate the final model with the correct resolution and normalized height
model = geo.GeoModel(bounds=bounds, resolution=res)
model.add_history(normed_hist)
model.clear_data()
model.compute_model()

p = geovis.volview(model)
p.show()

# Equivalently from library
model = gen.generate_normalized_model(hist, bounds=bounds, resolution=res)
print(model.xyz[np.argmax(model.data)])
geovis.volview(model).show()
