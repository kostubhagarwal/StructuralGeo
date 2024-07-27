""" 
Pytorch dataloader wrappers for specified GeoWord geological histories.

Requires:
"""

import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip
from structgeo.generation import GeologicalModelLoader

class GeoData3DStreamingDataset(Dataset):
    """ A Dataset wrapper for streaming  geological data from a generating yaml file and geowords.
    
    Parameters:
    config_yaml (str): Path to the yaml file containing the geological model generation configuration
    normalization_dir (str): Directory containing normalization statistics: mean_z.pt and std_dev_z.pt
    model_bounds (tuple): Bounds of the model in the form ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    model_resolution (tuple): Resolution of the model in the form (x_res, y_res, z_res)
    dataset_size (int): Number of samples in the dataset, arbirarily large since data is streamed
    device (str): Device to load the data to
    
    """
    def __init__(self, 
                 config_yaml: str,
                 normalization_dir: str = None,
                 model_bounds= ((-3840,3840),(-3840,3840),(-1920,1920)),    
                 model_resolution=(256,256,128), 
                 dataset_size=1_000_000, 
                 device='cpu'):
        self.model_generator = GeologicalModelLoader(config_yaml, model_bounds=model_bounds, model_resolution=model_resolution)
        self.normalization_dir = normalization_dir
        self.size = dataset_size
        self.device = device

        if normalization_dir:
            self.mean, self.std = load_normalization_stats(normalization_dir, device)
            # Check the last two dimensions of mean and std against the model resolution
            assert self.mean.shape == self.std.shape == self.model_generator.model_resolution[-1], \
            f"Normalization stats' {self.mean.shape} do not match the model's z resolution {self.model_generator.model_resolution[-1]}."

            self.normalize, self.denormalize = get_transforms(normalization_dir, data_dim=3, device=device)
      
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        model = self.model_generator.generate_models(1)[0] # Unwrap list of models
        model.fill_nans()
        data = model.get_data_grid()
        data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(self.device)
        if self.normalization_dir:
            data_tensor = self.normalize(data_tensor)
        return data_tensor
    
def load_normalization_stats(normalization_dir, device='cpu'):
    """ Load z-axis normalization statistics for the dataset from directory"""
    mean_path = os.path.join(normalization_dir, 'mean_z.pt')
    std_path = os.path.join(normalization_dir, 'std_dev_z.pt')
    
    # Assert that the mean and std dev files exist
    if not os.path.exists(mean_path):
        raise FileNotFoundError(f"Expected normalization file not found: {mean_path}")
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"Expected normalization file not found: {std_path}")
    
    mean = torch.load(mean_path).to(device)
    std = torch.load(std_path).to(device)
    return mean, std

def get_transforms(normalization_dir, data_dim , device='cpu', clamp_range=[-1, 20]):
    """ Get normalization and denormalization functions along Z axis
    
    Parameters:
    data_dir (str): Directory containing normalization stats in a 'normalization' subdirectory
    data_dim (int): Dimension of the data (2 or 3)
    device (str): Device to load normalization stats to
    clamp_range (list): Range to clamp the data to
    
    Returns:
    normalize (function): Function to normalize data
    denormalize (function): Function to denormalize data
    """    
    clamp_min, clamp_max = tuple(clamp_range)
    assert len(clamp_range) == 2, "clamp_range should contain exactly two values: [min, max]"    
    assert data_dim in [2, 3], "Data dimension should be 2 or 3 to specify 2d or 3d normalization"

    mean, std = load_normalization_stats(normalization_dir, device)
    
    if data_dim == 2:
        # Reshape the tensors to broadcast (Z,1) for 2d data
        mean.unsqueeze_(1)
        std.unsqueeze_(1)        

    def normalize(x):
        x = torch.clamp(x, clamp_min, clamp_max)
        return (x - mean.to(x.device)) / std.to(x.device)

    def denormalize(x):
        mean_res = mean.to(x.device)
        std_res = std.to(x.device)
        x = x * std_res + mean_res
        return torch.clamp(x, clamp_min, clamp_max)

    return normalize, denormalize

def compute_normalization_stats(dataset, batch_size, save_dir, device='cpu'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    sample = dataset[0]
    z = sample.shape[-1]
    

    
