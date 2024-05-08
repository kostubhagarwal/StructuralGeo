import os
import pickle

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
        if indexes:
            return max(indexes) + 1
        return 0

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

    def load_geo_model(self, file_name):
        """Load a GeoModel instance from a file."""
        file_path = os.path.join(self.base_dir, file_name)
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {file_path}")
        return model

    def save_history_models(self, models, lean=True):
        """Save a list of GeoModel instances.
        
        If lean is True, the models will be saved without the data attribute and only the 
        essential generating parameters (history, bounds, etc. )
        """
        for model in models:
            self.save_geo_model(model, lean=lean)

    def load_history_models(self):
        """Load all GeoModel instances from a directory."""
        models = []
        for filename in os.listdir(self.base_dir):
            if filename.endswith(".pkl"):
                models.append(self.load_geo_model(filename))
        return models