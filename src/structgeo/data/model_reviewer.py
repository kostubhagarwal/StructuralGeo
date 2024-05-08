import ipywidgets as widgets
import pyvista as pv
import structgeo.model.plot as geovis
from IPython.display import display, clear_output
from .file_manager import FileManager

class ModelReviewer:
    def __init__(self, generate_model_func, base_dir="../saved_models"):
        """
        Initialize the ModelReviewer with specific model generation and plotting functions.

        Parameters
        - generate_model_func: function that returns a new model instance
        - base_dir: directory where models will be saved
        """
        self.generate_model = generate_model_func
        self.base_dir = base_dir
        self.output = widgets.Output()
        self.plotter = pv.Plotter()     # type: ignore # 
        self.plotter.add_axes(line_width=5)
        self.fm = FileManager(base_dir=self.base_dir)  # Ensure FileManager is properly defined
        self.init_buttons()  # Initialize buttons

    def init_buttons(self):
        """Initialize the control buttons and setup event handlers."""
        self.save_button = widgets.Button(description="Save Model")
        self.discard_button = widgets.Button(description="Discard Model")
        self.save_button.on_click(self.save_action)
        self.discard_button.on_click(self.discard_action)
        self.button_box = widgets.HBox([self.save_button, self.discard_button])
        display(self.button_box)  # Display buttons below the plot
    
    def plot_model(self, model):
        with self.output:
            clear_output(wait=True)
            if hasattr(self.plotter, 'last_mesh'):
                self.plotter.remove_actor(self.plotter.last_mesh)  # Remove the last mesh if it exists

            mesh = geovis.get_mesh_from_model(model)
            color_config = geovis.get_color_config()
            self.plotter.last_mesh = self.plotter.add_mesh(mesh, scalars="values", **color_config)
            self.plotter.show(window_size=[600, 400])

    def refresh_model(self):
        self.current_model = self.generate_model()  # Store current model
        self.plot_model(self.current_model)

    def save_action(self, b):
        """Save the current model and refresh the review."""
        self.fm.save_geo_model(self.current_model)
        print("Model saved!")
        self.refresh_model()

    def discard_action(self, b):
        """Discard the current model and refresh the review."""
        print("Model discarded.")
        self.refresh_model()

    def start_review(self):
        display(self.output)  # Display the output widget which will contain everything
        self.refresh_model()  # Start the review by displaying the first model
