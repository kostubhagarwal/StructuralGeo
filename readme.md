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

