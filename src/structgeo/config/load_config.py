""" Main configuration file to load the dataset and model generation configuration. 

A default configuration file is provided in the repository, which points to a dataset directory
containing a YAML file, with optional dataset statistics in stats folder.
"""

import os
import json

def load_config(name='config_default.json'):
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
        
    # set a default stats directory to store and load normalization stats from  (add /stat/ to the dataset directory)
    config['stats_dir'] = os.path.join(dataset_dir, 'stats')
        
    return config
