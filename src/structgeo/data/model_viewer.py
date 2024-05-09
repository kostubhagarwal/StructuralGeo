import pyvista as pv
from structgeo.data.file_manager import FileManager  # Adjust the import path as necessary
from structgeo.plot import plot as geovis  # Adjust the import path as necessary
class ModelViewer:
    def __init__(self, base_dir="../database/"):
        """
        Initializes a model viewer that loads models from a specified directory and
        allows navigation using a slider widget in Pyvista.
        
        Parameters:
        - base_dir (str): Directory where models are stored.
        """
        self.base_dir = base_dir
        # Load all the models to view
        self.fm = FileManager(base_dir=self.base_dir)
        self.models = self.fm.load_all_models()
        self.curr_model = None 
        self.curr_mesh = None 

        # Initialize the PyVista plotter
        self.plotter = pv.Plotter()  # type: ignore
        self.plotter.add_axes(line_width=5)

        if self.models:
            # Initially display the first model
            self.update_plot(0)

        # Add a slider widget to navigate through models
        self.plotter.add_slider_widget(
            self.update_plot, [0, len(self.models) - 1], title="Model Index",
            fmt="%0.0f",
        )

        # Show the plotter window
        self.plotter.show()        

    def update_plot(self, model_index):
        """
        Callback function to update the plot based on the slider position.
        
        Parameters:
        - model_index (int): Index of the model to display.
        """
        model_index = int(model_index)  # Ensure the index is an integer
        if 0 <= model_index < len(self.models):
            if self.curr_mesh:
                self.plotter.remove_actor(self.curr_mesh)  # Remove the previous actor
                self.curr_model.clear_data() # type: ignore

            model = self.models[model_index]
            self.curr_model = model
            model.compute_model()  # Assuming there is a function to compute the model
            mesh = geovis.get_mesh_from_model(model)  # Assuming get_mesh() prepares the mesh
            color_config = geovis.get_color_config()  # Assuming there is a function to get color config

            # Add the new mesh to the plotter and store the actor for removal later
            self.curr_mesh = self.plotter.add_mesh(mesh, **color_config)
            self.plotter.render()  # Ensure the plotter re-renders the scene


if __name__ == "__main__":
    viewer = ModelViewer(base_dir="database")
