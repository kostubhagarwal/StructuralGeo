import os

from qtpy import QtCore, QtWidgets

from structgeo.filemanagement import FileManager


class FileManagerGUI:
    VIEWED_FILE_TYPES = ('.pkl', '.png', '.npy')
    
    def __init__(self, parent, file_manager):
        self.parent = parent
        self.fm : FileManager = file_manager
        self.base_dir = file_manager.base_dir
        # Create the file tree widget
        self.file_tree = QtWidgets.QTreeWidget(parent)
        self.file_tree.setHeaderLabels(["Models"])
        self.file_tree.itemSelectionChanged.connect(self.on_file_selected)
        
    def select_folder(self):
        """Open a dialog to select a folder and load models from the selected directory."""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self.parent, "Select Folder", "")
        if folder_path:
            self.base_dir = folder_path
            self.fm = FileManager(base_dir=self.base_dir)
            self.populate_file_tree()

    def populate_file_tree(self):
        """Populate the file tree with the directory structure and model files."""
        self.file_tree.clear()
        root_item = QtWidgets.QTreeWidgetItem(self.file_tree, [os.path.basename(self.base_dir)])
        self._populate_tree_widget(root_item, self.base_dir)
        self.file_tree.collapseAll()

    def _populate_tree_widget(self, parent_item, path):
        """Recursively populate the tree widget with the directory structure."""
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                tree_item = QtWidgets.QTreeWidgetItem(parent_item, [item])
                self._populate_tree_widget(tree_item, item_path)
            elif item_path.endswith(self.VIEWED_FILE_TYPES):
                tree_item = QtWidgets.QTreeWidgetItem(parent_item, [item])
                tree_item.setData(0, QtCore.Qt.UserRole, item_path)

    def on_file_selected(self):
        """Callback function when a file is selected in the tree."""
        selected_items = self.file_tree.selectedItems()
        
        if selected_items:
            item = selected_items[0]
            file_path = item.data(0, QtCore.Qt.UserRole)
            # Validate the file path as being a .pkl file
            if file_path and file_path.endswith(".pkl"):
                self.load_model(file_path)

    def load_model(self, file_path):
        """Load and display the selected model."""
        model = self.fm.load_geo_model(file_path)
        if model:
            self.parent.plotter.update_model(model)