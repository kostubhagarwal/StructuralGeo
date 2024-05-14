from pyvistaqt import QtInteractor
from qtpy import QtWidgets
import structgeo.plot as geovis

class ModelPlotter:
    def __init__(self, parent):
        self.parent = parent
        self.frame = QtWidgets.QFrame(parent)
        self.vlayout = QtWidgets.QVBoxLayout(self.frame)
        self.plotter = QtInteractor(self.frame)
        self.vlayout.addWidget(self.plotter.interactor)
        self.frame.setLayout(self.vlayout)
        self.curr_model = None  # Store the current model
        self.mesh = None  # Store the current mesh
        self.actors = []  # List to track all actors added to the plotter
        self.plotter.add_axes(line_width=5)
        self.plot_view_mode = self.plot_volume_view  # Default view mode
        self.color_config = geovis.get_plot_config()

    def update_plot(self, model):
        self.remove_all_actors()  # Remove all actors before updating the plot
        if self.curr_model is not None:
            self.curr_model.clear_data() # Wipe the model data
        self.curr_model = model  # Update the current model
        self.curr_model.compute_model() # Compute data for the model
        self.plot_view_mode()  # Update the plot with the new model
    
    def renormalize_height(self):
        if self.curr_model:
            self.curr_model.renormalize_height(auto = True)
            self.remove_all_actors()  # Remove all actors before updating the plot
            self.plot_view_mode()  # Update the plot with the new model

    def plot_volume_view(self):
        self.remove_all_actors()   # Remove all actors before updating the plot
        if self.curr_model:
            self.mesh = geovis.get_mesh_from_model(self.curr_model)
            a = self.plotter.add_mesh(self.mesh, **self.color_config)
            self.actors.append(a)
            self.plotter.render()

    def plot_orthslice_view(self):
        self.remove_all_actors()  # Remove all actors before updating the plot
        if self.curr_model:
            self.mesh = geovis.get_mesh_from_model(self.curr_model)
            a = self.plotter.add_mesh_slice_orthogonal(self.mesh, **self.color_config)
            self.actors.append(a)
            self.plotter.render()

    def plot_nslice_view(self, n=5, axis="x"):
        self.remove_all_actors()  # Remove all actors before updating the plot
        if self.curr_model:
            self.mesh = geovis.get_mesh_from_model(self.curr_model)
            slices = self.get_slices(n=n, axis=axis)
            a = self.plotter.add_mesh(slices, **self.color_config)
            self.actors.append(a)
            self.plotter.render()
            
    def get_slices(self, n=5, axis="x"):
        if self.mesh is not None:
            slices = self.mesh.slice_along_axis(n=n, axis=axis)
            return slices
        return None

    def plot_transformation_view(self):
        self.remove_all_actors()   # Remove all actors before updating the plot
        if self.curr_model:
            self.mesh = geovis.get_mesh_from_model(self.curr_model)

            a = self.plotter.add_mesh(self.mesh, **self.color_config)
            self.actors.append(a)
            actors = geovis.add_snapshots_to_plotter(self.plotter, self.curr_model, self.color_config['cmap'])
            self.actors.extend(actors)
            self.plotter.render()

    def remove_all_actors(self):
        """Remove all actors from the plotter."""
        for actor in self.actors:
            self.plotter.remove_actor(actor)
        self.plotter.clear_plane_widgets()
        
    def change_view_mode(self, mode):
        if mode == "Volume View":
            self.plot_view_mode = self.plot_volume_view
        elif mode == "OrthSlice View":
            self.plot_view_mode = self.plot_orthslice_view
        elif mode == "n-Slice View":
            n = self.parent.toolbar.n_spin_box.value()
            axis = self.parent.toolbar.axis_combo.currentText()
            self.plot_view_mode = lambda: self.plot_nslice_view(n=n, axis=axis)
        elif mode == "Transformation View":
            self.plot_view_mode = self.plot_transformation_view
        self.plot_view_mode()  # Update the view with the current model