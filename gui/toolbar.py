from qtpy import QtWidgets
import slicing_tool as sm
from structgeo.data import FileManager
import torch
from datetime import datetime

class ToolBarWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, plotter=None, file_manager=None):
        super(ToolBarWidget, self).__init__(parent)
        
        # Store references to plotter and file manager
        self.plotter = plotter
        self.file_manager: FileManager = file_manager
        
        # Create the main layout for the toolbar
        self.main_layout = QtWidgets.QHBoxLayout(self)
        
        # Initialize the combo box and button layouts
        self.init_button_layout()
        self.init_combo_box()
        
        # Set the main layout
        self.setLayout(self.main_layout)
        
        # Connect signals to slots (methods to handle button clicks)
        self.connect_signals()
        
    def init_combo_box(self):
        # Add dropdown menu for plotting types
        self.plotting_type_combo = QtWidgets.QComboBox(self)
        self.plotting_type_combo.addItems(["Volume View", 
                                           "OrthSlice View", 
                                           "n-Slice View", 
                                           "Transformation View", 
                                           "Categorical Grid View"])
        
        # Create a layout for the combo box and add it to the main layout
        self.combo_box_layout = QtWidgets.QHBoxLayout()
        self.combo_box_layout.addWidget(self.plotting_type_combo)
        self.main_layout.addLayout(self.combo_box_layout)
        
    def init_button_layout(self):
        # Create a layout for the buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)
        
        # Volume View buttons
        self.renormalize_button = QtWidgets.QPushButton("Renormalize Height", self)
        self.save_model_button = QtWidgets.QPushButton("Save Model", self)

        # n-Slice View buttons
        self.view_slices_button = QtWidgets.QPushButton("View Slices", self)
        self.save_slices_button = QtWidgets.QPushButton("Save Slices", self)        
        self.n_label = QtWidgets.QLabel("n-slices:", self)
        self.n_spin_box = QtWidgets.QSpinBox(self)
        self.n_spin_box.setRange(1, 100)  # Example range
        self.n_spin_box.setValue(5)  # Set initial value to 5
        self.axis_label = QtWidgets.QLabel("Axis:", self)
        self.axis_combo = QtWidgets.QComboBox(self)
        self.axis_combo.addItems(["x", "y", "z"])
        
        # Categorical Grid View buttons
        self.category_label = QtWidgets.QLabel("Category:", self)
        self.category_spin_box = QtWidgets.QSpinBox(self)
        self.category_spin_box.setRange(0, 1)          

        # List of all buttons and widgets
        self.all_widgets = [
            self.renormalize_button,
            self.save_model_button,
            self.view_slices_button,
            self.save_slices_button,
            self.n_label,
            self.n_spin_box,
            self.axis_label,
            self.axis_combo,
            self.category_label,
            self.category_spin_box,
        ]
        
        # Add buttons to the layout and hide them initially
        for widget in self.all_widgets:
            self.button_layout.addWidget(widget)
            widget.hide()
            
        self.update_toolbar("Volume View")
        
    def connect_signals(self):
        self.plotting_type_combo.currentIndexChanged.connect(self.on_plotting_type_changed)
        self.renormalize_button.clicked.connect(self.on_renormalize_clicked)
        self.save_model_button.clicked.connect(self.on_save_model_clicked)
        self.view_slices_button.clicked.connect(self.on_view_slices_clicked)
        self.save_slices_button.clicked.connect(self.on_save_slices_clicked)
        self.n_spin_box.editingFinished.connect(self.on_n_spin_box_finished)
        self.axis_combo.currentIndexChanged.connect(self.on_axis_combo_changed)
        self.category_spin_box.editingFinished.connect(self.on_category_changed)
                
    def update_toolbar(self, mode):
        # Hide all widgets
        for widget in self.all_widgets:
            widget.hide()
        
        # Show or hide buttons based on the mode
        if mode == "Volume View":
            self.renormalize_button.show()
            self.save_model_button.show()
            self.view_slices_button.show()
            self.save_slices_button.show()
        elif mode == "n-Slice View":
            self.view_slices_button.show()
            self.save_slices_button.show()
            self.n_label.show()
            self.n_spin_box.show()
            self.axis_label.show()
            self.axis_combo.show()
        elif mode == "Categorical Grid View":
            self.category_label.show()
            self.category_spin_box.show()
            
    def on_plotting_type_changed(self):
        mode = self.plotting_type_combo.currentText()
        self.update_toolbar(mode)
        QtWidgets.QApplication.processEvents()  # Force GUI update
        self.plotter.change_view_mode(mode)
                            
    def on_renormalize_clicked(self):
        self.plotter.renormalize_height()
        pass
        
    def on_save_model_clicked(self):
        # Save the model as an int8 tensor
        try:
            tensor = self.plotter.get_model_tensor()
        except ValueError as e:
            print(e)
            return
        # Prompt the user to select a save directory
        options = QtWidgets.QFileDialog.Options()
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory", options=options)
        
        if output_dir:
            # Get a unique filename using timestamp
            unique_id = datetime.now().strftime("%m%d%H%M%S")
            # Save the tensor as a .pt file
            torch.save(tensor, f"{output_dir}/model_{unique_id}.pt")
            print(f"Model saved as model_{unique_id}.pt")
        
    def on_n_spin_box_finished(self):
        # Refresh the plotter with the new n value
        self.plotter.change_view_mode("n-Slice View")

    def on_axis_combo_changed(self, index):
        # Refresh the plotter with the new axis
        self.plotter.change_view_mode("n-Slice View")
        
    def on_view_slices_clicked(self):
        slices = self.get_slices_from_plotter()
        # Plot the slices using matplotlib
        sm.plot_slices(slices)
    
    def on_save_slices_clicked(self):
        slices = self.get_slices_from_plotter()        
        # Open a directory selection dialog
        options = QtWidgets.QFileDialog.Options()
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory", options=options)
        
        if output_dir:
            # Save the slices as images and .npy files
            sm.save_slices_as_images(slices, output_dir, prefix=self.plotter.curr_model.name)
            sm.save_slices_as_npy(slices, output_dir)
    
    def get_slices_from_plotter(self):
        n = 32
        axis = self.axis_combo.currentText()
        model = self.plotter.curr_model
        # slices = sm.generate_slices(model, n, axis)
        slices = sm.generate_slices(model, n, "x") + sm.generate_slices(model, n, "y")
        
        return slices    
    
    def on_category_changed(self):
        # Update the plotter with the new category
        self.plotter.change_view_mode("Categorical Grid View")