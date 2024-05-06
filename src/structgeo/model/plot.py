import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pyvista as pv

class ColorMapConfig:
    """Configuration for discrete colormap with sentinel value (white) for non-rock areas."""
    def __init__(self, vmin=-1, vmax=10):
        self.vmin = vmin
        self.vmax = vmax
        num_colors = vmax - vmin    # Number of colors for the valid data range
        sentinel_color = np.array([1, 1, 1, 1])  # RGBA for non-rock sentinel color
        self.ticks = np.arange(vmin, vmax + 1)  # Adjusted ticks
        self.cmap = self.create_colormap(num_colors, sentinel_color)
        self.norm = self.create_norm()

    def create_colormap(self, num_colors, sentinel_color):
        # Use the viridis colormap for the valid data range
        viridis = plt.cm.get_cmap('viridis', num_colors)
        # Generate the colors for the valid data range
        colors = viridis(np.linspace(0, 1, num_colors))
        # Prepend the sentinel color
        all_colors = np.vstack((sentinel_color, colors))
        return ListedColormap(all_colors)

    def create_norm(self):
        # Define boundaries to accommodate sentinel at -1 and data values from vmin to vmax
        boundaries = np.linspace(self.vmin - .5, self.vmax + 0.5, self.vmax - self.vmin + 2)
        return BoundaryNorm(boundaries, len(boundaries) - 1)

color_config = ColorMapConfig(vmin=-1, vmax=10)
    
def volview(model, threshold=-0.5):
    mesh = get_mesh_from_model(model, threshold)
    
    # Create a plotter object
    plotter = pv.Plotter()      
    color_config = get_color_config()
    # Add the mesh to the plotter
    plotter.add_mesh(mesh, scalars="values", 
                     **color_config,
                     )
    # Show the plotter   
    plotter.show()    
    
def orthsliceview(model, threshold=-0.5):
    mesh = get_mesh_from_model(model, threshold)
    
    # Create a plotter object
    plotter = pv.Plotter()    
    color_config = get_color_config()     
    # Adding an interactive slicing tool
    plotter.add_mesh_slice_orthogonal(
                    mesh, scalars="values",
                    **color_config,
                    )    
    plotter.show() 
    
def nsliceview(model, n=5, axis="x", threshold=-0.5):
    mesh = get_mesh_from_model(model, threshold)
    slices = mesh.slice_along_axis(n=n, axis=axis)
    
    # Create a plotter object
    plotter = pv.Plotter()    
    color_config = get_color_config()     
    # Adding an interactive slicing tool
    plotter.add_mesh(slices, **color_config)  
    plotter.show()

def get_color_config(): 
    settings = {
        'cmap': color_config.cmap,
        'clim': (color_config.vmin, color_config.vmax),
        'scalar_bar_args': {
            'title': "Rock Type",
            'title_font_size': 16,
            'label_font_size': 10,
            'shadow': True,
            'n_labels': color_config.vmax - color_config.vmin + 1,
            'italic': True,
            'font_family': "arial",
        }
    }
    return settings
    
def get_mesh_from_model(model, threshold=-0.5):
    # Ensure the data is reshaped properly to match the grid dimensions 
    # X and Z seem to need to be swapped to match pyvista format when adding data values   
    grid = pv.StructuredGrid(model.X, model.Y, model.Z)
        
    # Set data to the grid
    values = model.data.reshape(model.X.shape)
    grid["values"] = values.flatten(order="F")  # Flatten the data in Fortran order 
    # Create mesh thresholding to exclude np.nan values or sentinel values
    mesh = grid.threshold(threshold, all_scalars=True) 
    return mesh


