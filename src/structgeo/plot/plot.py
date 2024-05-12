import pyvista as pv
import numpy as np

def get_plot_config(): 
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
            'n_labels': 5   # Reducing the number of labels for clarity
        }
    }
    return settings 
    
def volview(model, threshold=-0.5):
    mesh = get_mesh_from_model(model, threshold)
    
    # Create a plotter object
    plotter = pv.Plotter()       # type: ignore
    plot_config = get_plot_config()
    
    if np.all(np.isnan(model.data)):
        plotter.add_text("No data to show, all values are NaN.", font_size=20)
    else:
        # Add the mesh to the plotter
        plotter.add_mesh(mesh, scalars="values", 
                        **plot_config,
                        )
    _ = plotter.add_axes(line_width=5)
    return plotter    
    
def orthsliceview(model, threshold=-0.5):
    mesh = get_mesh_from_model(model, threshold)
    
    # Create a plotter object
    plotter = pv.Plotter()    # type: ignore
    color_config = get_plot_config()   
    if np.all(np.isnan(model.data)):
        plotter.add_text("No data to show, all values are NaN.", font_size=20)
    else:  
        # Adding an interactive slicing tool
        plotter.add_mesh_slice_orthogonal(
                        mesh, scalars="values",
                        **color_config,
                        )    
    _ = plotter.add_axes(line_width=5)
    return plotter    
    
def nsliceview(model, n=5, axis="x", threshold=-0.5):
    mesh = get_mesh_from_model(model, threshold)
    slices = mesh.slice_along_axis(n=n, axis=axis) # type: ignore
    
    # Create a plotter object
    plotter = pv.Plotter()    # type: ignore
    color_config = get_plot_config()     
    # Adding an interactive slicing tool
    if np.all(np.isnan(model.data)):
        plotter.add_text("No data to show, all values are NaN.", font_size=20)
    else:
        plotter.add_mesh(slices, **color_config)  
    _ = plotter.add_axes(line_width=5)
    return plotter
    
def get_mesh_from_model(model, threshold=-0.5):
    if model.data is None or model.data.size == 0:
        raise ValueError("Model data is empty or not computed, no data to show. Use compute model first.")
 
    grid = pv.StructuredGrid(model.X, model.Y, model.Z)
        
    # Set data to the grid
    values = model.data.reshape(model.X.shape)
    grid["values"] = values.flatten(order="F")  # Flatten the data in Fortran order 
    # Create mesh thresholding to exclude np.nan values or sentinel values
    mesh = grid.threshold(threshold, all_scalars=True) 
    return mesh


