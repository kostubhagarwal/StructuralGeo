import time as clock
import copy

import pyvista as pv
import numpy as np

import structgeo.model as geo
import structgeo.plot as geovis
from structgeo.generation import *



def single_plotter():
    # List of geological words to generate
    sentence = [BaseStrata(),Laccolith()]
    # Model resolution and boundse
    z = 64
    res = (2 * z, 2 * z, z)
    bounds = (
        BOUNDS_X,
        BOUNDS_Y,
        BOUNDS_Z,
    )  # Bounds imported from generation (geowords file)e

    hist = generate_history(sentence)
    start = clock.time()
    model = geo.GeoModel(bounds=bounds, resolution=res)
    model.add_history(hist)
    model.compute_model(normalize=True)
    finish = clock.time()
    print(f"Model computed in {finish-start:.2f} seconds.")

    # geovis.transformationview(model).show()e
    geovis.categorical_grid_view(model).show()
    print(model.get_history_string())

def get_ellipsoid_shaping_function(x_length, y_length, wobble_factor=1.):
    """ Organic Sheet Maker
    
    variance is introduced through 3 different fourier waves. x_var and y_var add ripple to the sheet thickness,
    while the radial_var adds a ripple around the edges of the sheet in the distance that it extends.
    
    The exponents (p) control the sharpness of the hyper ellipse:
    $$ 1 = (\frac{|z|}{d_z})^{p_z} + (\frac{|y|}{d_y})^{p_y} + (\frac{|x|}{d_x})^{p_x}$$
    
    This function has the z dimension normalized to 1, and the x and y dimensions are 
    normalized to the x_length and y_length. A sharp taper off at the edges is controlled with a 
    higher exp_z value.
    """
    # Make a fourier based modifier for both x and y
    fourier = rv.FourierWaveGenerator(num_harmonics=4, smoothness=1)
    x_var = fourier.generate()
    y_var = fourier.generate()
    radial_var = fourier.generate()
    amp = np.random.uniform(0.1, 0.2)*wobble_factor # unevenness of the dike thickness        
    exp_x = np.random.uniform(1.5,4) # Hyper ellipse exponent controls tapering sharpness
    exp_y = np.random.uniform(1.5,4) # Hyper ellipse exponent controls tapering sharpness
    exp_z = np.random.uniform(4,10)  # Hyper ellipse exponent controls tapering sharpness
    
    def func(x, y):
        # 3d ellipse with thickness axis of 1 and hyper ellipse tapering in x and y
        theta = np.arctan2(y, x)
        ellipse_factor = (1+ .6*radial_var(theta/(2*np.pi))) - np.abs(x/x_length)**exp_x - np.abs(y/y_length)**exp_y
        ellipse_factor = (np.maximum(ellipse_factor, 0))**(1/exp_z) 

        # The thickness modifier combines 2d fourier with tapering at ends
        return (1 + amp * x_var(x/X_RANGE)) * (1 + amp * y_var(y/X_RANGE)) * ellipse_factor
    
    return func

def get_hemi_function(wobble_factor=.1):
    """ Organic looking hemispheres
    
    The hemisphere coordinates xyz have been normalized to a simple hemisphere case where
    1=z^2+x^2+y^2 will give a default hemisphere, the purpose is to distort the default z surface
    """
    
    wf = wobble_factor
    fourier = rv.FourierWaveGenerator(num_harmonics=4, smoothness=1)
    x_var = fourier.generate()
    y_var = fourier.generate()
    exp_x = np.random.uniform(1.5,4) # Hyper ellipse exponent controls tapering sharpness
    exp_y = np.random.uniform(1.5,4) # Hyper ellipse exponent controls tapering sharpness
    exp_z = np.random.uniform(1.5,3)
    radial_var = fourier.generate()
    
    def func(x,y):
        x = (1+ wf*x_var(x))*x
        y = (1+ wf*y_var(y))*y
        r = 1+ .3*radial_var(np.arctan2(y,x)/(2*np.pi))
        inner = r**2 - np.abs(x)**exp_x - np.abs(y)**exp_y
        z_surf = np.maximum(0, inner) ** (1/exp_z)       
        return z_surf
    
    return func  
    
    
def process_plotter():
    bed = geo.Bedrock(0,1)
    sediment_word = Sediment()
    sediment = sediment_word.generate()
    
    hemi = geo.DikeHemispherePushed(
        origin=(0,0,1500),
        diam  =2000,
        height=300,
        minor_axis_scale=.5,
        rotation=0,
        value=5,
        upper=True,
        clip=False,
        z_function=get_hemi_function(wobble_factor=.05),
    )
    
    hist = [bed, sediment_word.generate(), sediment_word.generate(), hemi]
    # Model resolution and boundse
    z = 64
    res = (2 * z, 2 * z, z)
    bounds = (
        BOUNDS_X,
        BOUNDS_Y,
        BOUNDS_Z,
    )  # Bounds imported from generation (geowords file)
    model = geo.GeoModel(bounds=bounds, resolution=res)
    model.add_history(hist)
    model.compute_model(normalize=True)
    geovis.categorical_grid_view(model).show()

if __name__ == "__main__":
    single_plotter()
    
