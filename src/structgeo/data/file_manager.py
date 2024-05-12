import os
import dill as pickle
import structgeo.model as geo

class FileManager:
    """ A class to interface between GeoModel instances and pickled files on disk."""

    def __init__(self, base_dir="../saved_models"):
        self.base_dir = base_dir
        self.file_index = None
        
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
        
    def _file_save_string(self, model_index):
        """Format for saving model files."""
        return f"model_{model_index}.pkl"
   
    def save_geo_model(self, geo_model, save_dir, lean = True):
        """Save a GeoModel instance to a file.
        
        Parameters:
        geo_model: GeoModel instance
        lean: bool
        
        Description:
        Saves a Geomodel to disk in the base directory.
        
        If lean is True, the models will be saved without the data attribute and only the 
        essential generating parameters (history, bounds, etc. )
        """
        if self.file_index is None:
            self.file_index = self._get_initial_file_index()
               
        if lean:
            # Implement clear_data if needed to remove unnecessary large data
            geo_model.clear_data()
        file_path = os.path.join(save_dir, self._file_save_string(self.file_index))
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
            self.save_geo_model(model, self.base_dir, lean=lean)
    
    def walk_and_process_models(self, action_callback, *args, **kwargs):
        """Walk through directories and apply a callback to each model file."""
        print(f"Processing models in {self.base_dir}")
        for root, dirs, files in os.walk(self.base_dir):
            file_list = [os.path.join(root, file) for file in files if file.endswith(".pkl")]
            file_list.sort(key=lambda x: (os.path.dirname(x), int(os.path.basename(x).split('_')[-1].split('.')[0])))
            for file_path in file_list:
                action_callback(file_path, *args, **kwargs)
    
    def load_all_models(self):
        """Load all models."""
        models = []
        self.walk_and_process_models(lambda file_path: models.append(self.load_geo_model(file_path)))
        return models

    """ Model renewal functions for updating or pickled models with changes in the model class."""

    def renew_all_models(self, save_dir):
        """Process each model and save it to a new directory while preserving the directory structure.
        
        Used for updating models with new features or changes in the model class."""
        for root, dirs, files in os.walk(self.base_dir):
            file_list = [os.path.join(root, file) for file in files if file.endswith(".pkl")]
            file_list.sort(key=lambda x: (os.path.dirname(x), int(os.path.basename(x).split('_')[-1].split('.')[0])))
            for file_path in file_list:
                model = self.load_geo_model(file_path)
                model._validate_model_params()
                
                # Make any alterations needed here
                self.update_model_version(model)              
                # Create a new file path by replacing the base directory with the save directory
                new_file_path = os.path.join(save_dir, os.path.relpath(file_path, self.base_dir))
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                
                # Save the model to the new location
                with open(new_file_path, 'wb') as file:
                    pickle.dump(model, file)
                    print(f"Model saved to {new_file_path}")
               
    def update_model_version(self, model):
        """Update the model version to the latest version."""
        pass
                
    def custom_unpickle(file_path):
        """Temporary function to handle unpickling with a custom class remapping."""
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "structgeo.model.geo":
                    module = "structgeo.model"
                return super().find_class(module, name)
        with open(file_path, 'rb') as file:
            unpickler = CustomUnpickler(file)
            model = unpickler.load()
        return model
    
if __name__ == "__main__":
    file_manager = FileManager(base_dir="./database")
    file_manager.renew_all_models("./new_database")    
