## Project Description
StructuralGeo is a Python package for creating and visualizing synthetic structural geology data. The model data is handeled with a NumPy framework and the visualization is done with Pyvista.
### Project Installation
To install StructuralGeo to your Python environment, clone the repo first. Then go into the root StructuralGeo directory and then use the `setup.py` file in the root folder to install the package as editable. 

```bash
git clone https://github.com/eldadHaber/StructuralGeo
cd StructuralGeo
pip install -e .
```

After installation, the package can be imported into your Python environment with the following command:

```python
import structgeo
```
___
### Project Structure

The project is structured with the following directories:

| Directory          | Description                                                       |
|--------------------|-------------------------------------------------------------------|
| `database/`        | Contains a database of pickled GeoModels                          |
| `gui/`             | Contains a gui for viewing and slicing models                     |
| `code_examples/`   | Contains example Jupyter Notebooks that demonstrate package functionality |
| `model_generation/`| Contains notebooks that have been used to generate models for the database |
| `tests/`           | Contains some basic tests for the package                         |
| `src/structgeo/`   | Contains the main package code                                    |

### GUI Operation
Launch the gui by running the main.py. By default it opens a file tree viewer in the present working directory. Use the menu to select a database of .pkl models: `File->Select Models Folder`. 

The file tree will only show .pkl files and directories. Select a .pkl file to load it into the plotter window. Use the dropdown menu at the bottom to change between plotting modes.

In n-slice mode, select the number of slices and hit enter. Choose the slicing axis. Slices can be saved to a directory as both npy arrays and png files.

### Project Source

#### `data`
- **Description**: Contains data management functionality.
- **Components**:
  - **FileManager**: Manages loading and saving of pickled models from a base directory. Supports recursive operations and model updates.

#### `model`
- **Description**: Framework for creating and manipulating model data.
- **Components**:
  - **GeoModel**: Main class for creating a blank model, can be updated with geological history and visualized with PyVista.
  - **GeoProcess**: Handles the number-crunching part of the geological history of the model.
  - **history.py**: Start of a collection of higher abstraction geological history functions and helpers.
  - **util.py**: Contains helper functions for the model package.

#### `plot`
- **Description**: Contains visualization functionality for model data.
- **Components**:
  - **plot.py**: PyVista plotting library with basic plotting functions and standardized plotter setup.
  - **model_viewer.py**: Class for viewing a directory of models with a slider and a checkbox for slicing mode.
  - **model_generator.py**: GUI designed for Jupyter Notebook as a widget to view, save, or reject generated models and renormalize model height.

#### `probability`
- **Description**: Collection of functions and classes related to generating probability parameters of interest.

___
#### Jupyter Notebook Viewing

The visualization is handled with Pyvista which may require additional configuration for Jupyter Notebook to view the model iteractively. 

The type of visualization backend for Jupyter can be set to `static`, `html` or `trame`.

- `static` will render non-interactive plots quickly, recommended for quick testing.
- `html` will render interactive plots in the notebook but is slower to build the viewer than static. The model can be rotated and zoomed in the notebook.
- `trame` will render interactive plots in the notebook through a viewer with menu options. It takes a while to load and is slower to render than the other options.

The backend can be specified for all Pyvista plots at the top of the notebook with the following command:

```python
pyvista.set_jupyter_backend('static') # or 'html' or 'trame'
```

On an individual Pyvista plot that is returned by geovis package, the backend can be overridden with the following example command:

```python
p = geovis.volview(model)
p.window_size = window_size
p.add_title(title='Sedimentation Ontop of Bedrock Model', font_size=8)
p.show(jupyter_backend='static') 
```

To install the trame framework, use the following command:

```bash
pip install 'jupyterlab>=3' ipywidgets 'pyvista[all,trame]'   
```

If `trame` is not installed the jupyter backend should be set to static which will render non-interactive plots. See examples folder for implementation, or read more at the [Pyvista documentation](https://tutorial.pyvista.org/tutorial/00_jupyter/index.html).
