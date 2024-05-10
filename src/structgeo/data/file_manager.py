import os
import dill as pickle

class FileManager:

    def __init__(self, base_dir="../saved_models"):
        self.base_dir = base_dir
        self.file_index = self._get_initial_file_index()

    def _get_initial_file_index(self):
        """Determine the starting file index based on existing files in the directory."""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            return 0
        existing_files = [f for f in os.listdir(self.base_dir) if f.endswith('.pkl')]
        if not existing_files:
            return 0
        # Extract indexes from file names assuming the format 'model_<index>.pkl'
        indexes = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
        return max(indexes) + 1 if indexes else 0
    
    def sorted_by_index(files):
        """Sort a list of filenames based on the numerical index in their name."""
        sort_key = lambda x: int(x.split('_')[-1].split('.')[0])
        return sorted(files, key=sort_key)  # type: ignore

    def save_geo_model(self, geo_model, lean = True):
        """Save a GeoModel instance to a file.
        
        If lean is True, the models will be saved without the data attribute and only the 
        essential generating parameters (history, bounds, etc. )
        """       
        if lean:
            # Implement clear_data if needed to remove unnecessary large data
            geo_model.clear_data()
        file_path = os.path.join(self.base_dir, f"model_{self.file_index}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(geo_model, file)
        print(f"Model saved to {file_path}")
        self.file_index += 1

    def load_geo_model(self, file_path):
        """Load a GeoModel instance from a file."""
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def save_all_models(self, models, lean=True):
        """Save a list of GeoModel instances.
        
        If lean is True, the models will be saved without the data attribute and only the 
        essential generating parameters (history, bounds, etc. )
        """
        for model in models:
            self.save_geo_model(model, lean=lean)

    def load_all_models(self):
        """Load all GeoModel instances from a directory and its subdirectories, sorted by index."""
        models = []
        for root, dirs, files in os.walk(self.base_dir):
            print(f"Loading models from {root}")
            file_list = [os.path.join(root, file) for file in files if file.endswith(".pkl")]
            file_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for file_path in file_list:
                model = self.load_geo_model(file_path)
                models.append(model)
        return models