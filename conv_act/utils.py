#!/usr/bin/env python
import yaml

# CONFIG FILE    
def read_yaml(path):
    """
    Read a yaml file from a certain path.
    """
    stream = open(path, 'r')
    dictionary = yaml.safe_load(stream)
    return dictionary