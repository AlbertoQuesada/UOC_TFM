'''
T R A I N I N G   N N

Created on 11/04/2021
@author: Alberto Quesada
'''

### IMPORTS ###
import tensorflow as tf
#import tensorflow_addons as tfa
from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import json

### CLASSES ###
ahead = 5
class LastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, ahead, 1])

class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

#### DEFINES ###
# Export model to TensorFlow Lite using converter
def tflite_model(model, filename):
    # Initialize converter with model as arg
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Execute converter
    tflite_model = converter.convert()
    # Export to file
    with open(filename, 'wb') as f:
        f.write(tflite_model)

def compile_and_fit(model, window, epochs, patience=10):
    """
    Compile and fit the model
        
        Parameters
        ----------
        window: dataframe
            dataframe with the input data for training
        Returns
        -------
        model
            trained model
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping], verbose=0)
    return history

def last_baseline_model(window):
    """
    Perform baseline training: Y(t+1) = Y(t)
        
        Parameters
        ----------
        window: dataframe
            dataframe with the input data for training
        Returns
        -------
        model
            trained model
    """
    # Perform training
    baseline = LastBaseline()
    #baseline = Baseline(label_index=window.column_indices['pm_10'])
    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])
    # val_performance = {}
    # performance = {}
    # val_performance['Last_Base'] = baseline.evaluate(window.val)
    # performance['Last_Base'] = baseline.evaluate(window.test, verbose=0)
    # # Plot window
    # window.plot(baseline)
    # # Return
    # return baseline, val_performance, performance
    performance = ["Last baseline"]
    performance.extend(baseline.evaluate(window.val, verbose=0))
    performance.extend(baseline.evaluate(window.test, verbose=1))
    # val_target = window.val_df["pm_10"]
    # test_target = window.test_df["pm_10"]
    # val_abs_error = []
    # for i in range(len(val_target)-7):
    #     predictions = [val_target[i]]*5
    #     outputs = val_target[i+1:i+6]
    #     abs_error = (predictions-outputs).abs().mean()
    #     val_abs_error.append(abs_error)
    # test_abs_error = []
    # for i in range(len(test_target)-7):
    #     predictions = [test_target[i]]*5
    #     outputs = test_target[i+1:i+6]
    #     abs_error = (predictions-outputs).abs().mean()
    #     test_abs_error.append(abs_error)
    return baseline, performance#[pd.DataFrame(val_abs_error).abs().mean()[0], pd.DataFrame(test_abs_error).abs().mean()[0]]#performance

def repeat_baseline_model(window):
    """
    Perform baseline training: Y(t+1) = Y(t)
        
        Parameters
        ----------
        window: dataframe
            dataframe with the input data for training
        Returns
        -------
        model
            trained model
    """
    # Perform training
    baseline = RepeatBaseline()
    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])
    # val_performance = {}
    # performance = {}
    # val_performance['Repeat_Base'] = baseline.evaluate(window.val)
    # performance['Repeat_Base'] = baseline.evaluate(window.test, verbose=0)
    # # Plot window
    # window.plot(baseline)
    # # Return
    # return baseline, val_performance, performance
    performance = ["Repeat baseline"]
    performance.extend(baseline.evaluate(window.val, verbose=0))
    performance.extend(baseline.evaluate(window.test, verbose=0))
    return baseline, performance

def linear_model(window, epochs=200):
    """
    Perform linear training: y(t+1) = a*y(t)+b
        
        Parameters
        ----------
        window: dataframe
            dataframe with the input data for training
        Returns
        -------
        model
            trained model
    """
    # Define model architecture
    linear = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        # tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Flatten(),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(ahead*1,#*6,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([ahead, 1])#6
    ])
    # Perform training
    history = compile_and_fit(linear, window, epochs)
    # Evaluate model
    # val_performance = {}
    # performance = {}
    # val_performance['Linear'] = linear.evaluate(window.val)
    # performance['Linear'] = linear.evaluate(window.test, verbose=0)
    # # Plot window
    # window.plot(linear)
    # # Return
    # return linear, val_performance, performance
    performance = ["Linear"]
    performance.extend(linear.evaluate(window.val, verbose=0))
    performance.extend(linear.evaluate(window.test, verbose=1))
    return linear, performance

def dense_model(window, n_1=64, n_2=64, epochs=200):
    """
    Perform dense training
        
        Parameters
        ----------
        window: dataframe
            dataframe with the input data for training
        Returns
        -------
        model
            trained model
    """
    # Define model architecture
    dense = tf.keras.Sequential([
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        # tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Flatten(),
        # Shape => [batch, 1, dense_units]
        tf.keras.layers.Dense(n_1, activation='relu'),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(ahead*1,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([ahead, 1])
    ])
    # Perform training
    history = compile_and_fit(dense, window, epochs)
    # # Evaluate model
    # val_performance = {}
    # performance = {}
    # val_performance['dense'] = dense.evaluate(window.val)
    # performance['dense'] = dense.evaluate(window.test, verbose=0)
    # # Plot window
    # window.plot(dense)
    # # Performance
    # # Return
    # return dense, val_performance, performance
    performance = ["Dense"]
    performance.extend(dense.evaluate(window.val, verbose=0))
    performance.extend(dense.evaluate(window.test, verbose=1))
    return dense, performance

def multi_step_dense_model(window, n_1=64, n_2=64, epochs=200):
    """
    Perform multistep dense training
        
        Parameters
        ----------
        window: dataframe
            dataframe with the input data for training
        Returns
        -------
        model
            trained model
    """
    # Define model architecture
    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=n_1, activation='relu'),
        tf.keras.layers.Dense(units=n_2, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])
    # Perform training
    history = compile_and_fit(multi_step_dense, window, epochs)
    # Evaluate model
    val_performance = {}
    performance = {}
    val_performance['multi_step_dense'] = multi_step_dense.evaluate(window.val)
    performance['multi_step_dense'] = multi_step_dense.evaluate(window.test, verbose=1)
    # Plot window
    window.plot(multi_step_dense)
    # Return
    return multi_step_dense, val_performance, performance

def conv1D_model(window, filters=32, conv_width=10, n_1=32, epochs=200): #Conv_wind=lags
    """
    Perform conv 1D training
        
        Parameters
        ----------
        window: dataframe
            dataframe with the input data for training
        Returns
        -------
        model
            trained model
    """
    # Define model architecture
    conv_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -3:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(n_1, activation='relu', kernel_size=(3,)),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(ahead*3,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([ahead, 3])
    ])
    # Perform training
    history = compile_and_fit(conv_model, window, epochs)
    # Evaluate model
    # val_performance = {}
    # performance = {}
    # val_performance['conv'] = conv_model.evaluate(window.val)
    # performance['conv'] = conv_model.evaluate(window.test, verbose=0)
    # # Plot window
    # window.plot(conv_model)
    # # Return
    # return conv_model, val_performance, performance
    performance = ["Conv1D"]
    performance.extend(conv_model.evaluate(window.val, verbose=0))
    performance.extend(conv_model.evaluate(window.test, verbose=1))
    return conv_model, performance

def LSTM_model(window, n_1=32, return_sequences=True, epochs=200):
    """
    Perform LSTM training
        
        Parameters
        ----------
        window: dataframe
            dataframe with the input data for training
        Returns
        -------
        model
            trained model
    """
    # Define model architecture
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(n_1, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(ahead*1,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([ahead, 1])
    ])
    # Perform training
    history = compile_and_fit(lstm_model, window, epochs)
    # Evaluate model
    # val_performance = {}
    # performance = {}
    # val_performance['lstm'] = lstm_model.evaluate(window.val)
    # performance['lstm'] = lstm_model.evaluate(window.test, verbose=0)
    # # Plot window
    window.plot(lstm_model)
    # # Return
    # return lstm_model, val_performance, performance
    performance = ["LSTM"]
    performance.extend(lstm_model.evaluate(window.val, verbose=0))
    performance.extend(lstm_model.evaluate(window.test, verbose=1))
    return lstm_model, performance

def check_model_performance(performance, val_performance,lstm_model,n,l):
    """
    Check models performance
        
        Parameters
        ----------
        xx: xxx
            xxx
        Returns
        -------
        None
    """
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    # Check models performance
    x = np.arange(len(performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    metric_index = lstm_model.metrics_names.index('mean_absolute_error')
    val_mae = [v[metric_index] for v in val_performance.values()]
    test_mae = [v[metric_index] for v in performance.values()]
    plt.ylabel('mean_absolute_error [pm_2_5, normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(),
               rotation=45)
    _ = plt.legend()
    #plt.show()
    plt.savefig("lag_"+str(l)+"_neurons_"+str(n)+".png")
    for name, value in performance.items():
        print(f'{name:12s}: {value[1]:0.4f}')

def perform_training(window, lag):
    """
    Perform model training using LSTM NN architecture
        
        Parameters
        ----------
        window: dataframe
            dataframe with the input data for training
        Returns
        -------
        model
            trained model
    """
    n_list = [1,2,4,6,8,10,20,50]
    total_results = pd.DataFrame(columns=('model', 'val_mse', 'val_mae','test_mse', 'test_mae', 'lags', 'neurons'))
    for n in n_list:
        # Perform baseline training last
        print("    3.1 Baseline model last")
        last_baseline, last_performance = last_baseline_model(window)
        last_performance.extend([lag,n])
        #____________________________________________________________________________________ 
        # Perform baseline training repeat
        #print("    3.1 Baseline model repeat")
        #repeat_baseline, repeat_base_val_performance, repeat_base_performance = repeat_baseline_model(window)
        #____________________________________________________________________________________ 
        # Perform linear training
        print("    3.2 Linear model")
        linear, linear_performance = linear_model(window)
        linear_performance.extend([lag,n])
        #____________________________________________________________________________________ 
        # Perform dense training
        print("    3.3 Dense model")
        dense, dense_performance = dense_model(window, n_1 = n)
        dense_performance.extend([lag,n])
        #____________________________________________________________________________________ 
        # Perform multi step dense training
        #print("    3.4 Multi step dense model")
        #multistep_dense, multistep_dense_val_performance, multistep_dense_performance = multi_step_dense_model(window,n,n/2)
        #____________________________________________________________________________________ 
        # Perform conv training
        print("    3.5 1D CONV model")
        conv, conv_performance = conv1D_model(window, n_1 = n)
        conv_performance.extend([lag,n])
        #____________________________________________________________________________________ 
        # Perform LSTM training
        print("    3.6 LSTM model")
        lstm, lstm_performance = LSTM_model(window, n_1 = n)
        tflite_model(lstm,"model_2_5.tflite")
        lstm_performance.extend([lag,n])
    #     #____________________________________________________________________________________ 
    #     # Check models performance
    #     print("    3.7 Check models performance")
    #     performance = {**last_base_performance,**linear_performance,
    #                 **dense_performance,**conv_performance,**lstm_performance}
    #     val_performance = {**last_base_val_performance,**linear_val_performance,
    #                     **dense_val_performance,**conv_val_performance,**lstm_val_performance}
    #     print(performance)
    #     check_model_performance(performance, val_performance, lstm, n, lag)
    #     with open('lag_'+str(lag)+'_neurons_'+str(n)+'.json', 'w') as fp:
    #         json.dump(performance, fp)
    #     with open('val_lag_'+str(lag)+'_neurons_'+str(n)+'.json', 'w') as fp:
    #         json.dump(val_performance, fp)
    # #____________________________________________________________________________________ 
    # # Return
    # return performance, val_performance, lstm        
        results = pd.DataFrame(columns=('model', 'val_mse', 'val_mae','test_mse', 'test_mae', 'lags', 'neurons'))
        # results = results.append(pd.Series(last_performance, index=results.columns), ignore_index=True)
        # results = results.append(pd.Series(linear_performance, index=results.columns), ignore_index=True)
        # results = results.append(pd.Series(dense_performance, index=results.columns), ignore_index=True)
        # results = results.append(pd.Series(conv_performance, index=results.columns), ignore_index=True)
        results = results.append(pd.Series(lstm_performance, index=results.columns), ignore_index=True)
        total_results=total_results.append(results)
        #print(results)
    return total_results