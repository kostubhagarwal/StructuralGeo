import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotCrossSection(model):
    data = model.data
    X = model.X
    Y = model.Y
    Z = model.Z
    # Reshape the data back to the 3D grid
    data_reshaped = data.reshape(X.shape)
    # # Plotting a cross-section, for example at z index 10
    y_index = 10  # Change this index to view different vertical cross-sectional slices
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X[:, y_index, :], Z[:, y_index, :], data_reshaped[:, y_index, :], shading='auto')
    plt.colorbar()  # Show color scale
    plt.title(f'Vertical Cross-section at Y index {y_index}')
    plt.xlabel('X coordinate')
    plt.ylabel('Z coordinate')
    plt.show()

def plot3D(model):
    data = model.data
    X = model.X
    Y = model.Y
    Z = model.Z
    # Reshape the data back to the 3D grid
    data_reshaped = data.reshape(X.shape)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Choose indices for the cross sections
    y_index = 10  # for the X-Z plane cross-section
    x_index = 10  # for the Y-Z plane cross-section

    # Plot the X-Z cross-section as a texture on a Y constant plane
    XZ_x, XZ_z = X[:, y_index, :], Z[:, y_index, :]
    XZ_slice = data_reshaped[:, y_index, :]
    ax.plot_surface(XZ_x, np.full_like(XZ_x, Y[0, y_index, 0]), XZ_z, rstride=1, cstride=1, facecolors=plt.cm.viridis(XZ_slice / np.nanmax(XZ_slice)))

    # Plot the Y-Z cross-section as a texture on an X constant plane
    YZ_y, YZ_z = Y[x_index, :, :], Z[x_index, :, :]
    YZ_slice = data_reshaped[x_index, :, :]
    ax.plot_surface(np.full_like(YZ_y, X[x_index, 0, 0]), YZ_y, YZ_z, rstride=1, cstride=1, facecolors=plt.cm.viridis(YZ_slice / np.nanmax(YZ_slice)))

    # Set labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Color bar
    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=data_reshaped.min(), vmax=data_reshaped.max()))
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, orientation='vertical')
    cbar.set_label('Data value')

    # Set the aspect ratio of the plot
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    plt.title('3D Plot with X-Z and Y-Z Cross Section Images')
    plt.show()

def volview(model):
    data = model.data
    X = model.X
    Y = model.Y
    Z = model.Z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X, Y, Z, c=data, cmap='viridis')
    
    # Add color bar
    cbar = fig.colorbar(sc)
    cbar.set_label('Intensity')
    
    return fig, ax