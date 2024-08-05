import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

# File name conventions for normalization statistics
_mean_file = 'mean_z.pt'
_std_dev_file = 'std_dev_z.pt'

def compute_normalization_stats(stats_dir:str, dataset, batch_size:int, device='cpu'):
    """
    Compute and save normalization statistics for the z-axis of the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to compute statistics for.
    batch_size : int
        Number of samples per batch.
    stats_dir : str
        Directory where the computed statistics will be saved.
    device : str
        Device on which to perform computations.

    Notes
    -----
    This function iterates through the dataset to calculate mean and standard deviation
    for the z-axis. The results are saved in the specified directory.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)        
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
    os.makedirs(f"{stats_dir}", exist_ok=True)
    mean_path = os.path.join(stats_dir, _mean_file)
    std_path = os.path.join(stats_dir, _std_dev_file)
    torch.save(mean_z, mean_path)
    torch.save(std_dev_z, std_path)
    
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

def load_normalization_stats(stats_dir:str, device='cpu')->tuple:
    """
    Load z-axis normalization statistics for a dataset from the specified directory.

    Parameters
    ----------
    stats_dir : str
        The directory containing the normalization statistics files.
    device : str, optional
        The PyTorch device where the loaded tensors will be placed. Default is 'cpu'.

    Returns
    -------
    tuple
        A tuple containing two tensors: mean and standard deviation.

    Raises
    ------
    FileNotFoundError
        If the expected normalization files are not found in the provided directory.
    """
    mean_path = os.path.join(stats_dir, _mean_file)
    std_path = os.path.join(stats_dir, _std_dev_file)
    
    # Assert that the mean and std dev files exist
    if not os.path.exists(mean_path):
        raise FileNotFoundError(f"Expected normalization file not found: {mean_path}")
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"Expected normalization file not found: {std_path}")
    
    mean = torch.load(mean_path).to(device)
    std = torch.load(std_path).to(device)
    return mean, std

def get_transforms(stats_dir, data_dim: int , device='cpu', clamp_range:tuple=[-1, 30]):
    """
    Retrieve normalization and denormalization functions configured for specified data dimensions and range.

    Parameters
    ----------
    stats_dir : str
        Directory containing the statistics files for normalization.
    data_dim : int
        Dimension of the data to be transformed (2D or 3D).
    device : str
        PyTorch device where the normalization statistics are loaded.
    clamp_range : list
        Two-element list specifying the minimum and maximum clamping values for the normalization.

    Returns
    -------
    tuple
        A tuple containing two functions: `normalize` and `denormalize`.

    Raises
    ------
    AssertionError
        If the clamp range does not contain exactly two elements or if the data dimension is not 2 or 3.
    """  
    clamp_min, clamp_max = tuple(clamp_range)
    assert len(clamp_range) == 2, "clamp_range should contain exactly two values: [min, max]"    
    assert data_dim in [2, 3], "Data dimension should be 2 or 3 to specify 2d or 3d normalization"

    mean, std = load_normalization_stats(stats_dir, device)
    
    if data_dim == 2:
        # Reshape the tensors to broadcast (Z,1) for 2d data
        mean.unsqueeze_(1)
        std.unsqueeze_(1)        

    def normalize(x: torch.Tensor)->torch.Tensor:
        x = torch.clamp(x, clamp_min, clamp_max)
        return (x - mean.to(x.device)) / std.to(x.device)

    def denormalize(x: torch.Tensor)->torch.Tensor:
        mean_res = mean.to(x.device)
        std_res = std.to(x.device)
        x = x * std_res + mean_res
        return torch.clamp(x, clamp_min, clamp_max)

    return normalize, denormalize
