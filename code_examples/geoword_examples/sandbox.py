""" A simple demonstration sandbox for generating and visualizing geomodels."""

from structgeo.generation import * # Generation module contains all of the GeoWords and history generating functions
import structgeo.model as geo      # Model contains the GeoModel and GeoProcess classes, the mechanics of the modeling
import structgeo.plot as geovis    # Plot contains all the tools related to visualizing the geomodels

def direct_model():
    """ Demonstration of direct model building using GeoProcess class."""    
    # Now chain some geowords together to form a history
    # Hint: The infinite base-strata use 'Infinite' in their name
    geosentence = [InfiniteSedimentMarkov()] # Start with a base strata example
    
    # The generation/geohistgen.py file has some functions to convert a sentence to a history
    history = generate_history(geosentence)
    
    # This is done by calling the generate() method of each GeoWord in the sentence which samples 
    # from the GeoWord's distribution of GeoProcesses and RVs
    sample = InfiniteSedimentMarkov().generate()
    print(sample)
    
    bounds = (BOUNDS_X, BOUNDS_Y, BOUNDS_Z) # The generation import contains the intended bounds for the model
    z_res = 64                              # Set the resolution along z-axis
    resolution = (2*z_res, 2*z_res, z_res)  # Intended shape with bounds is 2x2x1, cubic voxels
    model = geo.GeoModel(bounds,resolution) # Initialize the model
    

def main():
    direct_model()




if __name__ == "__main__":
    main()

