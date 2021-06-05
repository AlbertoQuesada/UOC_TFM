'''
M A I N

Created on 11/04/2021
@author: Alberto Quesada
'''

#%%
### IMPORTS ###
from time import time
import imports as imp
import data_acquisition as da
import data_preparation as dp
import training_nn as tnn
import pandas as pd

#%%
### INIT STATEMENT ###
# Settings file
SETTINGS_FILE = "settings.yml"
SETTINGS = imp.import_settings(SETTINGS_FILE)

#%%
### MAIN ###
print("AIR QUALITY PREDICTION")

# Get current time
t_load_init = time()

print("1. LOADING DATA")
# Load data
raw_data = da.load_data(SETTINGS["filename"],SETTINGS["column_names"])

print("2. PREPARING DATA")
# Prepare data
w_list, l_list = dp.prepare_data(raw_data)

print("3. TRAINING MODEL")
# Perform training
results = pd.DataFrame(columns=('model', 'val_mse', 'val_mae','test_mse', 'test_mae', 'lags', 'neurons'))
for i,w in enumerate(w_list):
    results = results.append(tnn.perform_training(w, l_list[i]))
results.to_csv("results.csv")
# Execution time
t_load_end = time()
print("LOADING TIME:", 1000*(t_load_end-t_load_init), "milliseconds")