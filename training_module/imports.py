### IMPORT ###
import yaml
import json
#import tensorflow as tf


#### DEFINES ###
"""
This Function import settings
-------
inputs:
    filename: string - path to settings filename
-------
output:
    None
-------
"""
def import_settings(filename):
    with open(filename, 'r') as settings_file:
        try:
            settings = yaml.load(settings_file, Loader=yaml.FullLoader)
            print("Settings imported from {}".format(filename))
        except yaml.YAMLError as exc:
            print("Exception imports.py: try: yaml.load(settings_file). {}".format(str(exc)))
    return settings