from setuptools import find_packages, setup

setup(
    name="GeoGen",
    version="0.0.3",
    description="A package for creating, visualizing, and exporting 3D structural geology models. \
    Allows either user specified, or randomized generation of models.",
    packages=find_packages(where="src"),  # Search within 'src' directory for packages
    package_dir={"": "src"},  # Root package directory is 'src'
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyvista[all]",
        "ipywidgets",
        "ipykernel",
        "trame",
        "trame-vuetify",
        "trame-vtk",
        "tqdm",
        "PyDTMC",  # Added PyDTMC to the list of required packages
    ],
)
