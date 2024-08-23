"""
PyTorch DataLoader for streaming GeoWord geological histories. 
"""

import torch
from torch.utils.data import DataLoader, Dataset

# Two types of geological model generators provided
from structgeo.generation import MarkovGeostoryGenerator, YAMLGeostoryGenerator


class GeoData3DStreamingDataset(Dataset):
    """
    A Dataset wrapper for streaming geological data from a generating YAML file and geowords.

    Parameters
    ----------
    model_bounds : tuple
        Bounds of the model as ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    model_resolution : tuple
        Resolution of the model as (x_res, y_res, z_res).
    dataset_size : int
        The total number of samples in one epoch.
    device : str
        Torch device where data is loaded.
    """

    _GENERATOR_CLASS = MarkovGeostoryGenerator

    def __init__(
        self,
        model_bounds=((-3840, 3840), (-3840, 3840), (-1920, 1920)),
        model_resolution=(256, 256, 128),
        generator_config=None,
        dataset_size=1e6,
        device="cpu",
    ):
        self.model_generator = self._GENERATOR_CLASS(
            model_bounds=model_bounds,
            model_resolution=model_resolution,
            config=generator_config,
        )
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
