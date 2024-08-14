import os

import matplotlib.pyplot as plt
import pyvista as pv
from torch.utils.data import DataLoader
from tqdm import tqdm

import structgeo.plot as geovis
from structgeo.config import load_config
from structgeo.dataset import (
    GeoData3DStreamingDataset,
    compute_normalization_stats,
    get_transforms,
)
from structgeo.model import GeoModel

""" 
Load a default config pointing to a default dataset directory with yaml file.
"""


def dataset_test():
    """
    Check that the dataset can be loaded and a sample can be drawn from it.
    Check the conversion of a tensor back into a model for display.
    """
    config = load_config()
    yaml_loc = config["yaml_file"]
    bounds = ((-3840, 3840), (-3840, 3840), (-1920, 1920))
    resolution = (128, 128, 64)
    dataset = GeoData3DStreamingDataset(
        config_yaml=yaml_loc, model_bounds=bounds, model_resolution=resolution
    )

    # Draw a sample from the torch dataser
    sample = dataset[0]
    print(f"Datlaloader yields a sample of shape: {sample.shape}")
    print(sample)

    # Convert the tensor back into a model for display
    model = GeoModel.from_tensor(bounds=bounds, data_tensor=sample)
    geovis.volview(model).show()

    print("")


def loader_test():
    """
    Verify that the loader can be used to stream models from the generator.
    Display a sample of the models in a multiplotter window.
    """
    config = load_config()
    yaml_loc = config["yaml_file"]
    # Load the generator and model dimensions/bounds with computed stats
    bounds = ((-3840, 3840), (-3840, 3840), (-1920, 1920))
    resolution = (128, 128, 64)
    dataset = GeoData3DStreamingDataset(
        config_yaml=yaml_loc,
        dataset_size=1_000_000,  # 1 million models in loader, just for fun
        model_bounds=bounds,
        model_resolution=resolution,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    for i, batch in enumerate(tqdm(loader)):
        print(i, batch.shape)
        if i == 10:
            # Show a sample of the batch in mulitplotter window
            p = pv.Plotter(shape=(4, 4))
            for i, data in enumerate(batch):
                p.subplot(i // 4, i % 4)
                model = GeoModel.from_tensor(bounds=bounds, data_tensor=data)
                geovis.volview(model, threshold=-0.5, plotter=p)
            break

    # view the models
    p.show()
    print("")


def normalization_tests():
    """
    Verify normalization stats from dataset
    """
    config = load_config()
    yaml_loc = config["yaml_file"]
    # Load the generator and model dimensions/bounds with computed stats
    bounds = ((-3840, 3840), (-3840, 3840), (-1920, 1920))
    resolution = (64, 64, 32)
    dataset = GeoData3DStreamingDataset(
        config_yaml=yaml_loc,
        dataset_size=8000,
        model_bounds=bounds,
        model_resolution=resolution,
    )

    compute_normalization_stats(
        stats_dir=".", dataset=dataset, batch_size=8, device="cpu"
    )
    normalize, denormalize = get_transforms(stats_dir=".", data_dim=3)

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    for i, batch in enumerate(tqdm(loader)):
        print(i, batch.shape)
        if i == 10:
            # Show a sample of the batch in mulitplotter window
            p = pv.Plotter(shape=(4, 4))
            for i, data in enumerate(batch):
                i = i * 2
                p.subplot(i // 4, i % 4)
                data_normed = normalize(data)
                model = GeoModel.from_tensor(bounds=bounds, data_tensor=data_normed)
                geovis.volview(model, threshold=-10, plotter=p)
                p.add_title(f"Model {i//2} Normalized", font_size=8)

                i = i + 1
                p.subplot(i // 4, i % 4)
                # Denormalize the data so it looks normal
                data = denormalize(data_normed)
                model = GeoModel.from_tensor(bounds=bounds, data_tensor=data)
                geovis.volview(model, threshold=-0.5, plotter=p)
                p.add_title(f"Model {i//2} Denormalized", font_size=8)
            break

    # view the models
    p.show()
    print("")


if __name__ == "__main__":
    dataset_test()
    loader_test()
    normalization_tests()
