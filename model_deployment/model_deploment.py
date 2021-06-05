### IMPORTS ###
import bme280
import smbus2
from time import sleep, time
from datetime import datetime
from sds011 import *
import csv
import tensorflow as tf
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite

### CLASSES ###
# From TFlow documentation: 
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df, val_df, test_df,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

### INITIALIZE ###
# BME280 config
port = 1
address = 0x76
bus = smbus2.SMBus(port)
bme280.load_calibration_params(bus,address)
# SDS011 config
sds011 = SDS011("/dev/ttyUSB0")
sds011.sleep(sleep=True)
# Model deployment config
pm_2_5_lags = 10
pm_10_lags = 10
# Buffer config
pm_2_5_buffer = [[0 for col in range(3)] for row in range(pm_2_5_lags)]
pm_10_buffer = [[0 for col in range(5)] for row in range(pm_10_lags)]
# Scaling statistics
pm_2_5_inputs_mean = pd.read_csv("pm_2_5_mean.csv",index_col=0,header=None,sep=';')
pm_2_5_inputs_std = pd.read_csv("pm_2_5_std.csv",index_col=0,header=None,sep=';')
pm_10_inputs_mean = pd.read_csv("pm_10_mean.csv",index_col=0,header=None,sep=';')
pm_10_inputs_std = pd.read_csv("pm_10_std.csv",index_col=0,header=None,sep=';')
# Load TFLite model 
interpreter_2_5 = tf.lite.Interpreter("model_2_5.tflite")
interpreter_10 = tf.lite.Interpreter("model_10.tflite")

### MAIN ###
# Initialize inf loop
while True:
	# Get init time
    init_time = time()
	# Turn ON sds011 sensor
    sds011.sleep(sleep=False)
	# Wait 15s to request data
    sleep(15)
	# Request measure to the SDS011 sensor
    pm_2_5,pm_10 = sds011.query()
	# Turn OFF sds011 sensor
    sds011.sleep(sleep=True) 
	# Request measure to the BME280 sensor
    bme280_data = bme280.sample(bus,address)
    humidity = bme280_data.humidity
    pressure = bme280_data.pressure
    temperature = bme280_data.temperature
    # Generate daytime variable
    day = 24*60*60
    daily_product = (init_time+15) * (2 * np.pi / day)
	# Append to buffers and remove first
    pm_2_5_buffer.append([np.cos(daily_product),humidity,pm_2_5])
    pm_10_buffer.append([np.sin(daily_product),np.cos(daily_product),pressure,humidity,pm_10])
    pm_2_5_buffer.pop(0)
    pm_10_buffer.pop(0)
    # Scale data
    pm_2_5_inputs = pd.DataFrame(pm_2_5_buffer, columns=["Day_cos","humidity","pm_2_5"])
    pm_10_inputs = pd.DataFrame(pm_10_buffer, columns=["Day_sin","Day_cos","pressure","humidity","pm_10"])
    pm_2_5_inputs = pm_2_5_inputs[["humidity","pm_2_5","Day_cos"]]
    pm_10_inputs = pm_10_inputs[["pressure","humidity","pm_10","Day_sin","Day_cos"]]
    pm_2_5_inputs_scaled = (pm_2_5_inputs-pm_2_5_inputs_mean)/pm_2_5_inputs_std
    pm_10_inputs_scaled = (pm_10_inputs-pm_10_inputs_mean)/pm_10_inputs_std
    # Generate window
    pm_2_5_w = WindowGenerator(input_width=pm_2_5_lags, label_width=5, shift=5,
                               train_df=pm_2_5_inputs_scaled,val_df=pm_2_5_inputs_scaled,test_df=pm_2_5_inputs_scaled,label_columns=["pm_2_5"])
    pm_10_w = WindowGenerator(input_width=pm_10_lags, label_width=5, shift=5,
                              train_df=pm_10_inputs_scaled,val_df=pm_10_inputs_scaled,test_df=pm_10_inputs_scaled,label_columns=["pm_10"])
    # Allocate tensors.
    interpreter_2_5.allocate_tensors()
    interpreter_10.allocate_tensors()
    # Get input and output tensors.
    input_details_2_5 = interpreter_2_5.get_input_details()
    output_details_2_5 = interpreter_2_5.get_output_details()
    input_details_10 = interpreter_10.get_input_details()
    output_details_10 = interpreter_10.get_output_details()
    # Test model on random input data.
    input_data_2_5 = np.array(pm_2_5_w.train, dtype=np.float32)
    input_data_10 = np.array(pm_10_w.train, dtype=np.float32)
    interpreter_2_5.set_tensor(input_details_2_5[0]['index'], input_data_2_5)
    interpreter_10.set_tensor(input_details_10[0]['index'], input_data_10)
    interpreter_2_5.invoke()
    interpreter_10.invoke()
    output_data_2_5 = interpreter_2_5.get_tensor(output_details_2_5[0]['index'])
    output_data_10 = interpreter_10.get_tensor(output_details_10[0]['index'])
    output_data_2_5_unscaled = (output_data_2_5*pm_2_5_inputs_std.loc["pm_2_5"].values)+pm_2_5_inputs_mean.loc["pm_2_5"].values
    output_data_10_unscaled = (output_data_10*pm_10_inputs_std.loc["pm_10"].values)+pm_10_inputs_mean.loc["pm_10"].values
    print(datetime.now())
    print("predictions PM2.5: ",output_data_2_5_unscaled)
    print("predictions PM10: ",output_data_10_unscaled)
	# Open file and write data
    with open('/home/pi/Projects/data_acquisition/predictions_pm_2_5.csv','a') as out:
        writer = csv.writer(out)
        writer.writerow(output_data_2_5_unscaled)
    with open('/home/pi/Projects/data_acquisition/predictions_pm_10.csv','a') as out:
        writer = csv.writer(out)
        writer.writerow(output_data_10_unscaled)
	# Wait till next minute.
    time_spent = time()-init_time
    sleep(60-time_spent)