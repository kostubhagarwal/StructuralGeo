"""
A module for plotting views and visualization of GeoModel objects.
"""

import pyvista as pv
import numpy as np
import functools

def get_plot_config(n_colors=10): 
    """ Central color configurations and appearance settings for the plotter"""
    settings = {
        # Color map for the rock types
        'cmap': "gist_ncar",  # Vibrant color map to differentiate rock types
        # Scalar bar settings
        'scalar_bar_args': {
            'title': "Rock Type",
            'title_font_size': 16,
            'label_font_size': 10,
            'shadow': True,
            'italic': True,
            'font_family': "arial",
            'n_labels': 0,
            'vertical': True,
        }
        ,
    }
    return settings 

def setup_plot(model, plotter=None, threshold=-0.5):
    """ Common setup for all plot types """
    if plotter is None:
        plotter = pv.Plotter()
        
    if np.all(np.isnan(model.data)):
        plotter.add_text("No data to show, all values are NaN.", font_size=20)
        return plotter, None

    mesh = get_voxel_grid_from_model(model, threshold)
    unique_values = np.unique(model.data)
    # Filter out np.nan from the unique values
    unique_values = unique_values[~np.isnan(unique_values)]
    n_colors = unique_values.size
    plot_config = get_plot_config(n_colors=n_colors)
    
    # Convert values to strings for categorical data
    mesh['values'] = mesh['values'].astype(int).astype(str)
    
    return plotter, mesh, plot_config

def volview(model, threshold=-0.5, show_bounds=False, plotter=None):
    plotter, mesh, plot_config = setup_plot(model, plotter, threshold)
    if mesh is None:
        return plotter
    
    plotter.add_mesh(mesh, scalars="values", **plot_config, interpolate_before_map=False)
    plotter.add_axes(line_width=5)
    
    if show_bounds:
        plotter.show_bounds(
            grid='back',
            location='outer',
            ticks='outside',
            n_xlabels=4,
            n_ylabels=4,
            n_zlabels=4,
            xtitle='Easting',
            ytitle='Northing',
            ztitle='Elevation',
            all_edges=True,
        )
        
    flat_bounds = [item for sublist in model.bounds for item in sublist]
    bounding_box = pv.Box(flat_bounds)
    plotter.add_mesh(bounding_box, color="black", style="wireframe", line_width=1)
    
    return plotter

def orthsliceview(model, threshold=-0.5, plotter=None):
    plotter, mesh, plot_config = setup_plot(model, plotter, threshold)
    if mesh is None:
        return plotter
    
    plotter.add_mesh_slice_orthogonal(mesh, scalars="values", **plot_config)
    plotter.add_axes(line_width=5)
    return plotter

def nsliceview(model, n=5, axis="x", threshold=-0.5, plotter=None):
    plotter, mesh, plot_config = setup_plot(model, plotter, threshold)
    if mesh is None:
        return plotter
    
    slices = mesh.slice_along_axis(n=n, axis=axis)
    plotter.add_mesh(slices, scalars="values", **plot_config)
    plotter.add_axes(line_width=5)
    return plotter

def onesliceview(model, threshold=-0.5, plotter=None):
    plotter, mesh, plot_config = setup_plot(model, plotter, threshold)
    if mesh is None:
        return plotter
    
    skin = mesh.extract_surface()
    plotter.add_mesh_slice(mesh, **plot_config)
    plotter.add_mesh(skin, scalars='values', cmap=plot_config['cmap'], opacity=0.1, show_scalar_bar=False)
    plotter.add_axes(line_width=5)
    return plotter

def transformationview(model, threshold=None, plotter=None):
    plotter, mesh, plot_config = setup_plot(model, plotter, threshold)
    if mesh is None:
        return plotter

    final_mesh = get_voxel_grid_from_model(model, threshold)
    plotter.add_mesh(final_mesh, scalars="values", **plot_config, show_scalar_bar=False)
    plotter.add_axes(line_width=5)
    
    add_snapshots_to_plotter(plotter, model, plot_config['cmap'])
    return plotter

def add_snapshots_to_plotter(plotter, model, cmap):
    resolution = model.resolution
     # Calculate the offset to separate each snapshot
    # The offset is chosen based on the overall size of the model
    x_offset = model.bounds[0][1] - model.bounds[0][0]  # Width of the model along x
    
    # Remove first data time entry which is empty, add the final data time entry
    data_snapshots = np.concatenate((model.data_snapshots[1:], model.data.reshape(1, -1)), axis=0)
    
    # Reverse the snapshots
    mesh_snapshots = model.mesh_snapshots[::-1]
    data_snapshots = data_snapshots[::-1]
    
    actors = []    
    for i, (mesh_snapshot, data_snapshot) in enumerate(zip(mesh_snapshots, data_snapshots)):
        # Assuming snapshots are stored as Nx3 arrays
        # Reshape to 3D grid of points-- i.e. 4x4x4 grid of (x,y,z) points
        deformed_points = mesh_snapshot.reshape(resolution + (3,))
        grid = pv.StructuredGrid(deformed_points[..., 0] + (i+1) * x_offset * 1.3,  # Shift along x
                                deformed_points[..., 1], 
                                deformed_points[..., 2])
        # Set the same values to the new grid
        grid["values"] = data_snapshot.reshape(model.X.shape).flatten(order="F") # Assigning scalar values to the grid     
        # Add grid to plotter with a unique color and using the same scalar values
        a = plotter.add_mesh(grid, style='wireframe', scalars="values", cmap = cmap, line_width=1, show_scalar_bar=False)
        actors.append(a)
    
    return actors

def categorical_grid_view(model, threshold=None, text_annot = True):
    cfg = get_plot_config()
    
    def calculate_grid_dims(n):
        """ Calculate grid dimensions that are as square as possible. """
        sqrt_n = np.sqrt(n)
        rows = np.ceil(sqrt_n)
        cols = rows
        return int(rows), int(cols)
    
    grid = get_voxel_grid_from_model(model, threshold=threshold) # Get voxel grid
    cats = np.unique(grid['values']) # Find unique categories

    num_cats = len(cats)
    rows, cols = calculate_grid_dims(num_cats)
    
    p = pv.Plotter(shape=(rows, cols), border=False) # subplot square layout

    clim = [cats.min(), cats.max()] # Preset color limits for all subplots
    skin = grid.extract_surface() # Extract surface mesh for translucent skin

    for i, cat in enumerate(cats):
        row, col = divmod(i, cols)
        p.subplot(row, col)
        
        cat_mask = grid['values'] == cat # mask for a category
        category_grid = grid.extract_cells(cat_mask) # Pull only those cells from voxel grid

        # Plot the category cluster and a translucent skin for context
        p.add_mesh(skin, scalars='values', clim=clim, cmap = cfg['cmap'],
                opacity=0.2, show_scalar_bar=False)
        p.add_mesh(category_grid, scalars='values', clim=clim, cmap = cfg['cmap'],
                opacity=1.0, show_scalar_bar=False)


        if text_annot:
            p.add_text(f"Category {cat}", position='upper_left', font_size=10)

    # Link all views to synchronize interactions such as rotation and zooming
    p.link_views()
    return p
    
def get_mesh_from_model(model, threshold=None):
    """ Convert GeoModel data to a mesh grid of nodes for visualization     
    Total nodes is the same as data values, grid cells will be filled by interpolated rock type values
    """
    
    if model.data is None or model.data.size == 0:
        raise ValueError("Model data is empty or not computed, no data to show. Use compute model first.")
 
    grid = pv.StructuredGrid(model.X, model.Y, model.Z)
        
    # Set data to the grid
    values = model.data.reshape(model.X.shape)
    grid["values"] = values.flatten(order="F")  # Flatten the data in Fortran order 
    # Create mesh thresholding to exclude np.nan values or sentinel values
    mesh = grid.threshold(threshold, all_scalars=True) 
    return mesh

def get_voxel_grid_from_model(model, threshold = None):
    """ Convert GeoModel data to a voxel grid for visualization 
    Total cells is the same as data values, each cell given a discrete rock type value
    """
    if model.data is None or model.data.size == 0:
        raise ValueError("Model data is empty or not computed, no data to show. Use compute model first.")
    if not all(res > 1 for res in model.resolution):
        raise ValueError("Voxel grid requires a model resolution greater than 1 in each dimension.")
    
    # Create a padded grid with n+1 nodes and node spacing equal to model sample spacing    
    dimensions = tuple(x + 1 for x in model.resolution)
    spacing = tuple((x[1] - x[0])/(r-1) for x,r in zip(model.bounds, model.resolution))
    # pad origin with a half cell size to center the grid
    origin = tuple(x[0] - cs/2 for x,cs in zip(model.bounds, spacing))
       
    # Create a structured grid with n+1 nodes in each dimension forming n^3 cells
    grid = pv.ImageData(
        dimensions = dimensions,
        spacing = spacing,
        origin = origin,
    )    
    # Necessary to reshape data vector in Fortran order to match the grid
    grid['values'] = model.data.reshape(model.resolution).ravel(order='F')
    grid = grid.threshold(threshold, all_scalars=True)
    return grid