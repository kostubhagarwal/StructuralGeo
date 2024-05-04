import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D

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

def plotCrossSection(model, coord='y', slice_index=10):    
    # Ensure the coordinate is valid
    if coord not in ['x', 'y', 'z']:
        raise ValueError("Invalid coordinate specified; please use 'x', 'y', or 'z'.")
   
    # Mapping for two plot axes and data index
    axes = {
        'x': (model.Y, model.Z, 0),  # Y-Z plane
        'y': (model.X, model.Z, 1),  # X-Z plane
        'z': (model.X, model.Y, 2)   # X-Y plane
    }
    
    # Labels for the axes
    labels = {
        'x': ('Y coordinate', 'Z coordinate'),
        'y': ('X coordinate', 'Z coordinate'),
        'z': ('X coordinate', 'Y coordinate')
    }
          
    data = model.data
    # Reshape the data back to the 3D grid
    data_reshaped = data.reshape(model.X.shape)
    
    xlabel, ylabel = labels[coord]
    plane1, plane2, axis_index = axes[coord]
    coord1, coord2 = plane1.take(slice_index, axis=axis_index), plane2.take(slice_index, axis=axis_index)
    slice_data = data_reshaped.take(slice_index, axis=axis_index)

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(coord1, coord2, slice_data, cmap=color_config.cmap, norm=color_config.norm, shading='auto')
    plt.colorbar()  # Show color scale
    plt.title(f'Cross-section at {coord.upper()} index {slice_index}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def volview(model):
    data = model.data
    X = model.X
    Y = model.Y
    Z = model.Z
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot using the customized colormap and normalization
    sc = ax.scatter(X, Y, Z, c=data, cmap=color_config.cmap, norm=color_config.norm)
    
    # Add color bar with discrete steps
    cbar = fig.colorbar(sc, ticks=np.arange(-1, 11), extend='neither')  # Setting ticks at each integer
    cbar.set_label('Rock Type')
    
    # Setting labels for each axis
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    return fig, ax

def volmesh(model, threshold=-0.5):
    # Ensure the data is reshaped properly to match the grid dimensions 
    # X and Z seem to need to be swapped to match pyvista format when adding data values   
    grid = pv.StructuredGrid(model.Z, model.Y, model.X)
        
    # Set data to the grid
    values = model.data
    grid["values"] = values.flatten(order="F")  # Flatten the data in Fortran order 
    # Create mesh thresholding to exclude np.nan values or sentinel values
    mesh = grid.threshold(threshold, all_scalars=True) 
    
    # Create a plotter object
    plotter = pv.Plotter()      
   # Create a custom color bar using the colormap and clim defined in color_config
    sargs = dict(
    title = "Rock Type",
    title_font_size=16,
    label_font_size=10,
    shadow=True,
    n_labels= color_config.vmax - color_config.vmin + 1,
    italic=True,
    font_family="arial",
    )
    # Add the mesh to the plotter
    plotter.add_mesh(mesh, scalars="values", 
                     cmap=color_config.cmap, 
                     clim=(color_config.vmin, color_config.vmax), 
                     scalar_bar_args=sargs,
                     )
    # Show the plotter
    plotter.show()    

# TODO: Review this function
def plot3D(model):
    data = model.data
    X = model.X
    Y = model.Y
    Z = model.Z
    # Reshape the data back to the 3D grid
    data_reshaped = data.reshape(X.shape)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Choose indices for the cross sections
    y_index = 10  # for the X-Z plane cross-section
    x_index = 10  # for the Y-Z plane cross-section

    # Plot the X-Z cross-section as a texture on a Y constant plane
    XZ_x, XZ_z = X[:, y_index, :], Z[:, y_index, :]
    XZ_slice = data_reshaped[:, y_index, :]
    ax.plot_surface(XZ_x, np.full_like(XZ_x, Y[0, y_index, 0]), XZ_z, rstride=1, cstride=1, facecolors=plt.cm.viridis(XZ_slice / np.nanmax(XZ_slice)))

    # Plot the Y-Z cross-section as a texture on an X constant plane
    YZ_y, YZ_z = Y[x_index, :, :], Z[x_index, :, :]
    YZ_slice = data_reshaped[x_index, :, :]
    ax.plot_surface(np.full_like(YZ_y, X[x_index, 0, 0]), YZ_y, YZ_z, rstride=1, cstride=1, facecolors=plt.cm.viridis(YZ_slice / np.nanmax(YZ_slice)))

    # Set labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Color bar
    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=data_reshaped.min(), vmax=data_reshaped.max()))
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, orientation='vertical')
    cbar.set_label('Data value')

    # Set the aspect ratio of the plot
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    plt.title('3D Plot with X-Z and Y-Z Cross Section Images')
    plt.show()


