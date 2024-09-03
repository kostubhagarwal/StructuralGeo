import numpy as np
import torch
from pyvistaqt import QtInteractor
from qtpy import QtWidgets

import geogen.model as geo
import geogen.plot as geovis


class ModelPlotter:
    def __init__(self, parent):
        # ----- Layout ------
        self.parent = parent
        self.frame = QtWidgets.QFrame(parent)
        self.vlayout = QtWidgets.QVBoxLayout(self.frame)
        self.plotter = QtInteractor(self.frame)
        self.vlayout.addWidget(self.plotter.interactor)
        self.frame.setLayout(self.vlayout)

        # ----- Data ------
        self.curr_model: geo.GeoModel = None  # Store the current model
        self.resolution = None  # Model resolution override for faster plotting
        self.plotter.add_axes(line_width=5)
        self.plot_view_mode = self.plot_volume_view  # Default view mode
        self.plot_config = geovis.get_plot_config()

    def update_model(self, model):
        if self.curr_model is not None:
            del self.curr_model  # del prev model to free memory
        self.curr_model = model  # Update to the current passed model
        if self.resolution is not None:
            self.curr_model.resolution = self.resolution
        self.curr_model.compute_model()
        self.update_category_selector()
        self.plot_view_mode()  # Refresh the plotter with the new model

    def renormalize_height(self):
        if self.curr_model:
            self.curr_model.renormalize_height(auto=True)
            self.plot_view_mode()  # Update the plot with the changed model

    def plot_volume_view(self):
        self.remove_all_actors()  # Remove all actors before updating the plot
        if self.curr_model:
            geovis.volview(self.curr_model, plotter=self.plotter)
            self.plotter.render()

    def plot_orthslice_view(self):
        self.remove_all_actors()  # Remove all actors before updating the plot
        if self.curr_model:
            geovis.orthsliceview(self.curr_model, plotter=self.plotter)
            self.plotter.render()

    def plot_nslice_view(self, n=5, axis="x"):
        self.remove_all_actors()  # Remove all actors before updating the plot
        if self.curr_model:
            geovis.nsliceview(self.curr_model, n=n, axis=axis, plotter=self.plotter)
            self.plotter.render()

    def plot_transformation_view(self):
        self.remove_all_actors()  # Remove all actors before updating the plot
        if self.curr_model:
            geovis.transformationview(self.curr_model, plotter=self.plotter)
            self.plotter.render()

    def plot_categorical_grid_view(self, cat):
        self.remove_all_actors()  # Remove all actors before updating the plot
        if self.curr_model:

            grid = geovis.get_voxel_grid_from_model(self.curr_model, threshold=None)
            skin = grid.extract_surface()
            cat_mask = grid["values"] == cat
            # Ensure the mask is not empty
            if cat_mask.any():
                has_cat_data = True
            else:
                has_cat_data = False
                print(f"No data for category {cat}")
                return

            if has_cat_data:
                category_grid = grid.extract_cells(cat_mask)

            cats = np.unique(grid["values"])
            clim = [cats.min(), cats.max()]
            cfg = geovis.get_plot_config()
            cmap = cfg["cmap"]
            # Plot the category cluster and a translucent skin for context
            self.plotter.add_mesh(
                skin,
                scalars="values",
                clim=clim,
                cmap=cmap,
                opacity=0.2,
                show_scalar_bar=False,
            )
            if has_cat_data:
                self.plotter.add_mesh(
                    category_grid,
                    scalars="values",
                    clim=clim,
                    cmap=cmap,
                    opacity=1.0,
                    show_scalar_bar=False,
                )

            self.plotter.render()

    def update_category_selector(self):
        unique_values = np.unique(self.curr_model.data)
        unique_values = unique_values[~np.isnan(unique_values)]  # Filter out NaN values
        if unique_values.size > 0:
            min_value, max_value = int(unique_values.min()), int(unique_values.max())
            self.parent.toolbar.category_spin_box.setRange(min_value, max_value)
            self.parent.toolbar.category_spin_box.setValue(min_value)  # Set to the first category by default
        else:
            self.parent.toolbar.category_spin_box.setRange(0, 0)  # No valid categories
            self.parent.toolbar.category_spin_box.setValue(0)

    def remove_all_actors(self):
        """Remove all actors from the plotter."""
        self.plotter.clear_actors()
        self.plotter.clear_plane_widgets()

    def change_view_mode(self, mode):
        self.remove_all_actors()
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
        elif mode == "Categorical Grid View":
            cat = self.parent.toolbar.category_spin_box.value()
            self.plot_view_mode = lambda: self.plot_categorical_grid_view(cat)

        self.plot_view_mode()  # Update the view with the current model

    def get_model_tensor(self):
        if self.curr_model:
            np_array = self.curr_model.get_data_grid()
            tensor = torch.tensor(np_array, dtype=torch.int8)
            return tensor
        else:
            raise ValueError("No model to get tensor from")
