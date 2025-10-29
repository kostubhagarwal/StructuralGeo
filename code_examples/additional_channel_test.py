import pyvista as pv
from torch.utils.data import DataLoader
import geogen.plot as geovis  # Plot contains all the tools related to visualizing the geomodels
from geogen.dataset import GeoData3DStreamingDataset
from geogen.dataset.add_channel import add_channel
import geogen.model as geo

def main():
     dataloader()

def dataloader():
    bounds = ((-3840, 3840), (-3840, 3840), (-1920, 1920))
    resolution = (128, 128, 64)
    dataset = GeoData3DStreamingDataset(model_bounds=bounds, model_resolution=resolution)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    batch = next(iter(loader))
    batch_shape = batch.shape
    print(f"Dataloader yields a sample of shape: {batch_shape}")
    model = geo.GeoModel.from_tensor(bounds=bounds, data_tensor=batch[0, 0, ...])
    geovis.volview(model, show_bounds=True).show()

if __name__ == "__main__":
    main()