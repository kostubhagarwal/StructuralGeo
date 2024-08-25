import geogen.generation as gen
import geogen.plot as geovis
from geogen.model import GeoModel

# Geo sentence fromed of geowords
sentence = [gen.InfiniteBasement() , gen.CoarseRepeatSediment(), gen.FourierFold(), gen.SingleRandSediment(), gen.FineRepeatSediment()]
hist = gen.generate_history(sentence) 
res = (128,128,64)
bounds = ((-3840,3840),(-3840,3840),(-1920,1920)) 

def generate_model():
    # Generate a randomized history from geowords, one single sample
    hist = gen.generate_history(sentence)    
    # Generate a model
    model = GeoModel(bounds = bounds, resolution = res)
    model.add_history(hist)
    model.compute_model(normalize = True)
    return model

model = generate_model()
geovis.volview(model, show_bounds=True).show()