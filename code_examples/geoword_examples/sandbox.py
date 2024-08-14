""" A simple demonstration sandbox for generating and visualizing geomodels."""

import pyvista as pv

from structgeo.generation import * # Generation module contains all of the GeoWords and history generating functions
import structgeo.model as geo      # Model contains the GeoModel and GeoProcess classes, the mechanics of the modeling
import structgeo.plot as geovis    # Plot contains all the tools related to visualizing the geomodels

def main():
    # direct_model_generation_demo()
    geo_sentence_batch_sampling()
    summary()
   
def direct_model_generation_demo():
    """ Demonstration of direct model building using GeoProcess class."""    
    # Now chain some geowords together to form a history
    # Hint: The infinite base-strata use 'Infinite' in their name, the provide a base foundation for the model
    geosentence = [InfiniteSedimentMarkov(), FourierFold(), SingleDikeWarped(), CoarseRepeatSediment()]
        
    # All GeoWords have a generate() method that returns a packaged snippet of history
    sample = InfiniteSedimentMarkov().generate()
    
    # We can inspect each one to get an idea of what they do
    for geoword in geosentence:
        print(f"Generating Sample from {geoword.__class__.__name__}")
        print(geoword.generate())
        print("\n")
        
    # Now we can chain them together to form a history
    history = [geoword.generate() for geoword in geosentence]
       
    # The history is the instruction set for the GeoModel, fetch a model with a helper
    # function. It is convenient to fix the resolution at 2x2x1 ratio, cubic voxels
    # Defining only the z-axis resolution, the x and y axes are derived from this
    model = get_empty_model(z_axis_res=64)
    
    # Now with a model and a history, we can build the model
    model.add_history(history)
    model.compute_model()
    print(model.data)
    visualize_model_demo(model)
    
    # There is an issue to be addressed, the model is not filled in well.
    # An automatic normalization scheme has been implemented to address this 
    # in the generation module.
    model = generate_normalized_model(hist = history, 
                              bounds = (BOUNDS_X, BOUNDS_Y, BOUNDS_Z), 
                              resolution= (128,128,64) )
    visualize_model_demo(model)
    
    # There is also a visualization to show categorical chunks of the model
    p = geovis.categorical_grid_view(model)
    p.show()
    
def get_empty_model(z_axis_res=64):
    bounds = (BOUNDS_X, BOUNDS_Y, BOUNDS_Z) # The geowords import from structgeo.generation import * contains
                                            # the intended bounds for the geowords
    x = z_axis_res
    resolution = (2*x, 2*x, x)              # Intended shape with bounds is 2x2x1 ratio, cubic voxels
    model = geo.GeoModel(bounds,resolution) # Initialize the model
    return model
    
def visualize_model_demo(model):
    """ Demonstration of visualizing a geomodel."""    
    # Now we can visualize the model, there are a few options:    
    # geovis.volview(model, show_bounds=True)
    # geovis.orthsliceview(model)
    # geovis.nsliceview(model, n=10)
    # geovis.onesliceview(model)
    
    # All of these return a PyVista plotter object, or optionally a plotter can be passed in
    # As a 'plotter = ' argument. This allows for modes like subplotting, etc.    
    p = geovis.volview(model, show_bounds=True)
    p.show()
    
    # Now using passed plotter to view them all:
    p = pv.Plotter(shape=(2,2))
    p.subplot(0,0)
    geovis.volview(model, plotter=p)
    p.subplot(0,1)
    geovis.orthsliceview(model, plotter=p)
    p.subplot(1,0)
    geovis.nsliceview(model, n=10, plotter=p)
    p.subplot(1,1)
    geovis.onesliceview(model, plotter=p)
    
    p.show()

def geo_sentence_batch_sampling():
    """ Demonstration of batch sampling of geo sentences     
    A multi-plotter PyVista window offering viewing of GeoWord stories in 3D.

        Shortcut keys:
        - 'r': Refresh the samples.
        - '1': View the samples in volume mode.
        - '2': View the samples in orthogonal slice mode.
        - '3': View the samples in n-slice mode.
        - '4': View the samples in one-slice mode.  

        Plotter Parameters:
        - sentence (list): A list of GeoWords to generate histories from
        - bounds (tuple): The bounds of the model in the form ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        - res (tuple): The resolution of the model in the form (nx, ny, nz)
        - n_samples (int): The number of samples to generate and plot. Plotter defaults to square grid layout.
    """
    # A helper object to sample geosentences is provided in the plotting module:
    geosentence = [InfiniteSedimentMarkov(), FourierFold(), SingleDikeWarped(), CoarseRepeatSediment()]
    bounds = (BOUNDS_X, BOUNDS_Y, BOUNDS_Z)
    res = (64,64,32) # Lower resolution can be beneficial for speed when assessing many samples
    viewer = geovis.GeoWordPlotter(geosentence, bounds, res, n_samples=16)

def summary():
    # To summarize generation and testing via GeoWords:
    
    # 1. A reference guide to GeoProcesses to work with is in /icons/docs/process-icons.drawio-poster.pdf
    
    # 2. Premade GeoWords are found in /generation/geowords.py
    
    # 3. GeoWords can be chained together in a list:
    geosentence = [InfiniteSedimentUniform(), SimpleFold(), SingleDikeWarped(), CoarseRepeatSediment()]
    
    # 4. The sentence can be sampled into a concrete history:
    history = [geoword.generate() for geoword in geosentence]
    # Alternatively use the function from generation module:
    history = generate_history(geosentence)
    
    # 5. The history is added to a model and computed with height normalization to generate raw data:
    model = generate_normalized_model(hist = history,
                                      bounds = (BOUNDS_X, BOUNDS_Y, BOUNDS_Z), # Defined from in geowords.py
                                      resolution= (128,128,64) )
    
    # 6. Various visualization options are available in plot module:
    geovis.volview(model, show_bounds=True).show()
    

if __name__ == "__main__":
    main()

