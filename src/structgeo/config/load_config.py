""" Main configuration file to load the dataset and model generation configuration. 

A default configuration file is provided in the repository, which points to a dataset directory
containing a YAML file, with optional dataset statistics in stats folder.
"""

import json
import os


def load_config(name='config_default.json'):
    """ Load the configuration file and resolve relative paths.
    
    Parameters:
    name (str): Name of the configuration file to load. Default is 'config_default.json'. File should
    be located in the same directory as this script. The json should contain the following keys:
    
      -"dataset_dir": "generation/simple_dataset"
      
    The dataset_dir should contain a YAML file with the model generation parameters.   
    
    Returns:
    A dictionary containing the configuration parameters.
    - dataset_dir: str
    - yaml_file: str
    """
    
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Define the path to the configuration file
    config_path = os.path.join(dir_path, name)
    
    # Load the configuration file
    with open(config_path, 'r') as file:
        config = json.load(file)
 
    # Resolve relative paths
    dataset_dir = os.path.join(dir_path, config['dataset_dir'])   
    config['dataset_dir'] = dataset_dir
    
    # Search for YAML files in the dataset directory
    yaml_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.yaml', '.yml'))]
    if yaml_files:
        config['yaml_file'] = os.path.join(dataset_dir, yaml_files[0])
        print("YAML file found and loaded:", yaml_files[0])
    else:
        print("No YAML file found in the dataset directory.")
        
    return config
