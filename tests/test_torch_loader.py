import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from structgeo.dataloader import GeoData3DStreamingDataset
from structgeo.model import GeoModel
import structgeo.plot as geovis
from torch.utils.data import DataLoader

import torch

yaml_loc = 'C:/Users/sghys/Summer2024/StructuralGeo/src/structgeo/generation/grammar_map.yml' 
def model_loader_test():
    bounds = ((-3840,3840),(-3840,3840),(-1920,1920))
    resolution = (128,128,64)
    dataset = GeoData3DStreamingDataset(config_yaml = yaml_loc, model_bounds=bounds, model_resolution=resolution) 
    
    # Draw a sample from the torch dataser
    sample = dataset[0]
    print(sample.shape)
    print(sample)
    
    # Convert the tensor back into a model for display
    model = GeoModel.from_tensor(bounds = bounds, data_tensor=sample)
    geovis.volview(model).show()

    print('')
    
def model_norm_testing():
    bounds = ((-3840,3840),(-3840,3840),(-1920,1920))
    resolution = (128,128,64)
    dataset = GeoData3DStreamingDataset(config_yaml = yaml_loc, dataset_size=800, model_bounds=bounds, model_resolution=resolution)
    save_dir = 'C:/Users/sghys/Summer2024/StructuralGeo/tests/normed' 
    compute_normalization_stats(dataset, batch_size=8, save_dir=save_dir, device='cpu')
    
    
def compute_normalization_stats(dataset, batch_size, save_dir, device='cpu'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)        
    sample = dataset[0]
    z = sample.shape[-1]  
    
    tensor_mu_acc = torch.zeros(z)
    tensor_x_squared_acc = torch.zeros(z)
    
    n_batches = len(loader)
    print(f"Iterating over {n_batches} batches to compute normalization statistics")

    # Using tqdm to display progress
    for batch in tqdm(loader, total=n_batches, desc="Computing stats"):
        batch = batch.to(device)  # Ensure the batch is on the correct device
        tensor_mu_acc += batch.mean(dim=(0, 1, 2, 3), keepdim=False)
        tensor_x_squared_acc += (batch**2).mean(dim=(0, 1, 2, 3), keepdim=False)

    mean_z = tensor_mu_acc / n_batches
    std_dev_z= torch.sqrt(tensor_x_squared_acc / n_batches - mean_z**2)
    
    # Save the mean and std dev tensors as vectors to be used for normalization
    os.makedirs(f"{save_dir}/normalization", exist_ok=True)
    torch.save(mean_z, f"{save_dir}/normalization/mean_z.pt")
    torch.save(std_dev_z, f"{save_dir}/normalization/std_dev_z.pt")
    
    # Expand into a 2d matrix to use with imshow
    mean_z_matrix = mean_z.unsqueeze(0).expand(64,-1)
    std_dev_z_matrix = std_dev_z.unsqueeze(0).expand(64,-1)

    # Rotate the tensors by 90 degrees CCW (equivalent to a transpose followed by a flip on the last dimension)
    rotated_mean_z = torch.flip(mean_z_matrix.T, [0])
    rotated_std_dev_z = torch.flip(std_dev_z_matrix.T, [0])

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(rotated_mean_z.numpy(), cmap='gray')
    axs[0].set_title("Mean Z")
    axs[1].imshow(rotated_std_dev_z.numpy(), cmap='gray')
    axs[1].set_title("Std Dev Z")
    plt.show()
    
if __name__ == '__main__':
    # model_loader_test()
    model_norm_testing()