"""
PyTorch DataLoader for streaming GeoWord geological histories. This module includes a custom dataset class that 
streams geological model data using a generative model and YAML configuration. I

Requires:
- A GeoModelGenerator object capable of generating unlimited geological models.
- A YAML configuration file specifying generation parameters.
"""
import torch
from torch.utils.data import DataLoader, Dataset

from structgeo.generation import GeoModelGenerator


class GeoData3DStreamingDataset(Dataset):
    """
    A Dataset wrapper for streaming geological data from a generating YAML file and geowords.

    Parameters
    ----------
    config_yaml : str
        Path to the YAML configuration file for geological model generation.
    model_bounds : tuple
        Bounds of the model as ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    model_resolution : tuple
        Resolution of the model as (x_res, y_res, z_res).
    dataset_size : int
        The total number of samples in one epoch.
    device : str
        Torch device where data is loaded.
    """
    def __init__(self, 
                 config_yaml: str,
                 model_bounds= ((-3840,3840),(-3840,3840),(-1920,1920)),    
                 model_resolution=(256,256,128),  # Default bounds and resolution are paired to have 30m resolution
                 dataset_size= 1e6, 
                 device='cpu'):
        self.model_generator = GeoModelGenerator(config_yaml, model_bounds=model_bounds, model_resolution=model_resolution)
        self.device = device
        self.size = dataset_size
                     
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        model = self.model_generator.generate_model()
        model.fill_nans()
        data = model.get_data_grid()
        data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(self.device)
                    
        return data_tensor
        
