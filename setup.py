from setuptools import find_packages, setup

setup(
    name="StructuralGeo",
    version="0.0.2",
    description="A package for structural geology visualization and analysis",
    packages=find_packages(
        where="src"
    ),  # Search within 'src' directory for packages
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
