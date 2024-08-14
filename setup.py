from setuptools import find_packages, setup

setup(
    name="StructuralGeo",
    version="0.0.2",
    description="A package for structural geology visualization and analysis",
    packages=find_packages(
        where="src"
    ),  # Search within 'structgeo' directory for packages
    package_dir={"": "src"},  # Root package directory is 'structgeo'
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
    ],
)
