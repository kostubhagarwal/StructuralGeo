import ipywidgets as widgets
import pyvista as pv
import structgeo.plot.plot as geovis
from IPython.display import display, clear_output
from .file_manager import FileManager

class ModelReviewer:
    def __init__(self, generate_model_func, base_dir="../saved_models", show_history=True):
        """
        Initialize the ModelReviewer with specific model generation and plotting functions.

        Parameters
        - generate_model_func: function that returns a new model instance
        - base_dir: directory where models will be saved
        """
        self.generate_model = generate_model_func
        self.base_dir = base_dir
        self.output = widgets.Output()
        self.plotter = pv.Plotter(notebook = True)     # type: ignore # 
        self.plotter.add_axes(line_width=5)
        self.fm = FileManager(base_dir=self.base_dir)  
        self.show_history = show_history

    def init_buttons(self):
        """Initialize the control buttons and setup event handlers."""
        self.save_button = widgets.Button(description="Save Model")
        self.discard_button = widgets.Button(description="Discard Model")
        self.exit_button = widgets.Button(description="Exit Review")
        self.save_button.on_click(self.save_action)
        self.discard_button.on_click(self.discard_action)
        self.exit_button.on_click(self.exit_action)
        self.button_box = widgets.HBox([self.save_button, self.discard_button, self.exit_button])
        display(self.button_box)  # Display buttons below the plot
        
    def plot_model(self):
        with self.output:
            mesh = geovis.get_mesh_from_model(self.current_model)
            color_config = geovis.get_plot_config()
            clear_output(wait=True)  # Clear the output before starting to plot the new model
            if hasattr(self.plotter, 'last_mesh'):
                self.plotter.remove_actor(self.plotter.last_mesh)  # Remove the last mesh if it exists
         
            self.plotter.last_mesh = self.plotter.add_mesh(mesh, scalars="values", **color_config)  
            # INFO: For some outrageous reason the PyVista plotter needs to be shown twice or it will
            # lag by one render and show the previous mesh. This is sadly the workaround for now.   
            self.plotter.show(window_size=[600, 400], jupyter_backend='static')
            clear_output(wait=True)
            self.plotter.show(window_size=[600, 400], jupyter_backend='static')

    def refresh_model(self):
        new_model = self.generate_model()  # Generate new model
        self.current_model = new_model  # Update current model
        self.plot_model()  # Plot the new model  
        if (self.show_history):
            with self.output:     
                print(self.current_model.get_history_string())

    def save_action(self, b):
        """Save the current model and refresh the review."""
        self.fm.save_geo_model(self.current_model)  # Save the current model
        with self.output:
            print(f"Model saved to {self.base_dir}")
            self.refresh_model()  # Refresh to get a new model displayed

    def discard_action(self, b):
        """Discard the current model and refresh the review."""
        print("Model discarded.")
        self.refresh_model()  # Refresh to get a new model displayed
    
    def exit_action(self, b):
        """ Exit the review, close the plotter, and remove all UI components. """
        self.plotter.close()  # Close the plotter to free up resources
        self.output.close()  # Close the output widget to remove it from the display
        self.button_box.close()  # Close the button box widget to remove it from the display
        print("Review exited.")

    def start_review(self):
        self.init_buttons()  # Initialize buttons
        self.refresh_model()  # Start the review by displaying the first model
        display(self.output)  # Display the output widget which will contain everything
    