import pyvista as pv
from torch.utils.data import DataLoader
import geogen.plot as geovis  # Plot contains all the tools related to visualizing the geomodels
from geogen.dataset import GeoData3DStreamingDataset
from geogen.dataset.add_channel import add_channel
import geogen.model as geo

def main():
     dataloader()

def dataloader():
    bounds = ((0, 1.6e3), (0, 1.6e3), (0, 0.8e3))
    resolution = (32, 32, 16)
    dataset = GeoData3DStreamingDataset(model_bounds=bounds, model_resolution=resolution)
    # loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    print(f"Dataset shape: {dataset.__getitem__(0).shape}")
    # batch = next(iter(loader))
    # batch_shape = batch.shape
    # print(f"Dataloader yields a sample of shape: {batch_shape}")
    model = geo.GeoModel.from_tensor(bounds=bounds, data_tensor=dataset.__getitem__(0)[1, ...])
    geovis.volview(model, show_bounds=True).show()

if __name__ == "__main__":
    main()