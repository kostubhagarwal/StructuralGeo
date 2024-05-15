        
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from structgeo.data import FileManager
from datetime import datetime

def generate_slices(model, n, axis):    
    # Convert axis text to the corresponding axis index
    axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
    
    def get_single_layer_slices(data, n=5, axis=0):
        """Slice a numpy array to get single-layer slices along a specified axis."""
        slices = []
        axis_size = data.shape[axis]
        step = max(1, (axis_size / n))
        # Make index selction taking floor of index steps
        index_selection = np.floor(np.arange(0, axis_size, step)).astype(int)        
        
        for i in index_selection:
            if axis == 0:
                slice_data = data[i, :, :]
            elif axis == 1:
                slice_data = data[:, i, :]
            elif axis == 2:
                slice_data = data[:, :, i]
            
            # Rotate the slice 90 degrees CCW
            slice_data = np.rot90(slice_data)
            
            slices.append(slice_data)
        return slices
    
    # Fetch the current model data
    if model is not None:
        data = model.get_data_grid()
        if data is not None:
            # Get slices from the numpy array data
            slices = get_single_layer_slices(data, n=n, axis=axis_index)      
            # Fill the NaNs with a sentinel of -1
            slices = [np.nan_to_num(slice, nan=model.EMPTY_VALUE) for slice in slices]     
            return slices        
       
        else:
            print("Model data is None.")
    else:
        print("No current model available.")
        
def plot_slices(slices, max_cols=8):
    # Determine the number of rows and columns
    num_slices = len(slices)
    num_cols = min(num_slices, max_cols)
    num_rows = (num_slices + max_cols - 1) // max_cols  # Equivalent to ceil(num_slices / max_cols)

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = np.array(axes)  # Ensure axes is an array for consistent indexing

    # Flatten axes array for easier iteration in case of a single row or column
    if num_rows == 1 or num_cols == 1:
        axes = axes.flatten()

    for i, slice_data in enumerate(slices):
        print(f"Slice {i} shape: {slice_data.shape}")
        ax = axes.flat[i] if num_slices > 1 else axes
        ax.imshow(slice_data, cmap='viridis')
        ax.set_title(f"Slice {i}")
        ax.axis('off')  # Hide the axis

    # Hide any remaining empty subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes.flat[j])

    plt.tight_layout()
    plt.show()
    
def save_slices_as_images(slices, output_dir, prefix="slice"):
    """Save slices as PNG images."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, slice_data in enumerate(slices):
        image_path = os.path.join(output_dir, f"{prefix}_{timestamp}_{i}.png")
        # Convert slice data to uint8 for image saving
        slice_image = (255 * (slice_data - np.min(slice_data)) / np.ptp(slice_data)).astype(np.uint8)
        Image.fromarray(slice_image).save(image_path)
    print(f"Saved {len(slices)} slices as images in {output_dir}.")

def save_slices_as_npy(slices, output_dir, prefix="slice"):
    """Save slices as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, slice_data in enumerate(slices):
        npy_path = os.path.join(output_dir, f"{prefix}_{timestamp}_{i}.npy")
        np.save(npy_path, slice_data)
    print(f"Saved {len(slices)} slices as .npy files in {output_dir}.")
        
if __name__ == "__main__":
    # Testing the slicing tool
    fm = FileManager(base_dir="C:/Users/sghys/2024 Summer Work/StructuralGeo/database/faulted_models")
    models = fm.load_all_models()
    
    model = models[10]
    model.compute_model()
    
    # Generate slices of the model data
    slices = generate_slices(model, n=32, axis="x")
    if slices is not None:
        print("Slices generated successfully.")
    else:
        print("Failed to generate slices.")
        
    print(slices)
        

    plot_slices(slices)
    
    # Save slices to file
    save_dir = "C:/Users/sghys/2024 Summer Work/test_slicing"    