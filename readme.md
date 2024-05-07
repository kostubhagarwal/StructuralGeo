## Project Description
StructuralGeo is a Python package for creating and visualizing synthetic structural geology data. The model data is handeled with a NumPy framework and the visualization is done with Pyvista.
#### Project Installation
To install StructuralGeo to your Python environment, clone the repo and then use the `setup.py` file in the root folder. 

```bash
git clone https://github.com/yourusername/StructuralGeo.git
cd StructuralGeo
pip install -e .
```

After installation, the package can be imported into your Python environment with the following command:

```python
import structgeo
```

##### Jupyter Notebook Viewing

The visualization is handled with Pyvista which requires additional configuration for Jupyter Notebook to view the model iteractively. 

To install the required framework, use the following command:

```bash
pip install 'jupyterlab>=3' ipywidgets 'pyvista[all,trame]'   
```

If `trame` is not installed the jupyter backend should be set to static which will render non-interactive plots. To activate interactive plots set the backend to `trame` at the import header. See examples folder for implementation, or read more at the [Pyvista documentation](https://tutorial.pyvista.org/tutorial/00_jupyter/index.html).

