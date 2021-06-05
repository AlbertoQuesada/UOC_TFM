'''
D A T A   P R E P A R A T I O N

Created on 11/04/2021
@author: Alberto Quesada
'''

### IMPORT ###
# FeatureSelection as fsm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tracemalloc
from functools import reduce
from datetime import datetime, timedelta
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from pyod.models.knn import KNN
from time import time

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

    def plot(self, model=None, plot_col='pm_10', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col}')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)
            if n == 0:
                plt.legend()
        plt.xlabel('Time [min]')
        #plt.show()

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

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

#### DEFINES ###
def check_timeseries_gaps(data):
    """
    Check if time series dataframe contains gaps in time:

        Parameters
        ----------
        data: dataframes
            time series dataframe
        Returns
        -------
        None
    """
    # Take the diff and check if it is greater than 40ms
    new_data = data.reset_index(inplace=False)
    new_data["gap"] = (new_data["timestamp"].diff()).dt.seconds >= 61
    gaps = new_data[new_data["gap"]==True]
    if len(gaps) > 0:
        print(gaps)
        print('check_timeseries_gaps - Number can not be greater than 0. Value is ' + str(len(gaps)))
        #raise ValueError('check_timeseries_gaps - Number can not be greater than 0. Value is ' + str(len(gaps)))
    else:
        # Show details
        print("        - There are no GAPS in data")
    return new_data["gap"]

def any_is_null(x):
    """
    Check if list contains missing values:

        Parameters
        ----------
        x: list
            list to identify missing values
        Returns
        -------
        None
    """
    return any(pd.isnull(x))

def impute_missing_values_KNN(data):
    """
    Impute missing values in dataframe using KNN algorithm:

        Parameters
        ----------
        data: dataframes
            time series dataframe
        Returns
        -------
        dataframe
            dataframe with no missing values
    """
    # Impute missing values using KNN algorithm
    column_names = data.columns
    index = data.index
    imputer = KNNImputer()
    data = pd.DataFrame(imputer.fit_transform(data), columns=column_names, index=index)
    # Return
    return data

def check_missing_values(data):
    """
    Check if time series dataframe contains missing values:

        Parameters
        ----------
        data: dataframes
            time series dataframe
        Returns
        -------
        dataframe
            dataframe with no missing values
    """
    # Missing values identification
    is_null_data = data.apply(any_is_null)
    # Check point
    if any(is_null_data==True):
        data = impute_missing_values_KNN(data)
        print('There are missing values in data: ', is_null_data.to_string())
    else:
        # Show details
        print("        - There are no missing values in data")
    # Return
    return data

def evaluate_main_frequencies(data, feature):
    """
    Evaluate main frequencies from variable

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        Returns
        -------
        dataframe
            dataframe with new features
    """
    # Evaluate target frequencies
    fft = tf.signal.rfft(data[feature])
    f_per_dataset = np.arange(0, len(fft))

    n_samples_h = len(data[feature])
    hours_per_year = 25*60*60*24*365.2524
    years_per_dataset = n_samples_h/(hours_per_year)

    f_per_year = f_per_dataset/years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    #plt.show()
    #plt.ylim(0, 400000)
    #plt.xlim([0.1, max(plt.xlim())])
    #plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    #_ = plt.xlabel('Frequency (log scale)')
    # Return
    return data

def datetime_feature_extraction(data):
    """
    Feature extraction from datetime

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        Returns
        -------
        dataframe
            dataframe with new features
    """
    # Date time feature extraction
    timestamp_s = data.index.values.astype(np.int64) // 10 ** 9
    day = 24*60*60
    daily_product = timestamp_s * (2 * np.pi / day)
    data["Day_sin"] = pd.DataFrame(np.sin(daily_product),index=data.index)
    data["Day_cos"] = pd.DataFrame(np.cos(daily_product),index=data.index)
    # Return
    return data

def remove_constant_columns(data):
    """
    Remove constant columns from dataframe:

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        Returns
        -------
        dataframe
            dataframe without constant columns
    """
    # Remove constant columns
    data = data.loc[:, (data != data.iloc[0]).any()]
    # data = data.loc[:,data.apply(pd.Series.nunique) != 1]
    # Show details
    print("        - Data without constant columns: {} rows and {} columns".format(data.shape[0],data.shape[1]))
    # Return
    return data

def remove_constant_columns_threshold(data, threshold=0.01):
    """
    Remove constant columns with threshold from dataframe:

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        threshold: float
            indicates the variance threshold to be considered as constant
        Returns
        -------
        dataframe
            dataframe without constant columns
    """
    # Remove constant columns
    column_names = data.columns
    index = data.index
    constant_filter = VarianceThreshold(threshold)
    constant_filter.fit(data)
    # Get constant columns
    constant_columns = [column for column in data.columns if column not in data.columns[constant_filter.get_support()]]
    non_constant_columns = [column for column in data.columns if column in data.columns[constant_filter.get_support()]]
    # Show details
    if len(constant_columns) == 0:
        print("        - There are no constant columns in data")
    else:
        print("        - There are {} constant columns in data: {}".format(len(constant_columns),constant_columns))
    # Remove constant columns
    data = pd.DataFrame(constant_filter.transform(data), columns=non_constant_columns,index=index)
    print("        - Data without constant columns: {} rows and {} columns".format(data.shape[0],data.shape[1]))
    # Return
    return data

def remove_outliers(data):
    """
    Remove outliers from dataframe using KNN algorithm:

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        Returns
        -------
        dataframe
            dataframe without constant columns
    """
    # Remove outliers with KNN algorithm
    clf = KNN() # Initialize KNN instance
    clf.fit(data) # Fit model to the data
    predicted_class = clf.predict(data) # Identify outliers
    # Show details    
    if any(predicted_class==1):
        print("        - There are {} outliers in data".format(list(predicted_class).count(1)))
        # Remove outliers
        data["outlier"] = predicted_class
        data = data[data["outlier"]==0]#.interpolate(method='cubic')
        #data.index = dataIndex
        data = data.drop(["outlier"],axis=1)
    else:
        print("        - There are no outliers in data")
    # Return
    return data#.reset_index(drop=True)

def resample_time_series(data, frequency='1S'):
    """
    Resample time series to desired frequency:

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        frequency: string
            represents desired frequency. Default is 0.1 seconds
        Returns
        -------
        dataframe
            dataframe resampled to derised frequency
    """
    # Resample time series
    data = data.resample(frequency,label='right',closed='right').mean()
    # Show details
    print("        - Resampled data: {} rows and {} columns".format(data.shape[0],data.shape[1]))
    # Return
    return data

def perform_feature_selection_sklearn(data, target_feature='target', n_features=10):
    """
    Perform feature selection using sklearn:

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        Returns
        -------
        dataframe
            dataframe with the optimal set of features
    """
    # Perform feature selection
    bestfeatures = SelectKBest(score_func=f_regression, k='all')
    fit = bestfeatures.fit(data.drop(target_feature, inplace=False, axis=1),data[target_feature])
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data.drop(target_feature, inplace=False, axis=1).columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Features','Score']
    largest_scores = featureScores.nlargest(n_features,'Score')
    # Show details
    print(largest_scores)
    # Filter features
    data = data[largest_scores["Features"] + [target_feature]]
    # Show details
    print("        - Data with optimal features: {} rows and {} columns".format(data.shape[0],data.shape[1]))
    # Return
    return data

def perform_feature_selection_growing_inputs(data, target_feature='target', coeff_threshold=0.1):
    """
    Perform feature selection with growing inputs method:

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        Returns
        -------
        dataframe
            dataframe with the optimal set of features
    """
    # Perform feature selection
    corrs = data.corr()
    print(corrs)
    corrs = corrs[target_feature]
    corrs = corrs.iloc[(-corrs.abs()).argsort()]
    # Show details
    print(corrs)
    # Filter data
    data=data.drop(corrs[corrs.abs()<coeff_threshold].index, inplace=False, axis=1)
    # Show details
    print("        - Data with optimal features: {} rows and {} columns".format(data.shape[0],data.shape[1]))
    # Return
    return data

def scale_data_minmax(data):
    """
    Scale all features using Min and Max:

        Parameters
        ----------
        data: dataframe
            dataframe with the data
        Returns
        -------
        dataframe
            dataframe with the features scaled.
    """
    # Preprocess min and max
    minimum = data.min()
    maximum = data.max()
    # Scale the data
    data = (data-minimum)/(maximum-minimum)
    # Show details
    new_data=data.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=new_data)
    _ = ax.set_xticklabels(data.keys(), rotation=90)
    plt.show()
    # Return
    return data, minimum, maximum

def scale_data_meanstd(data, train_df, val_df, test_df):
    """
    Scale all features using Mean and Std:

        Parameters
        ----------
        train_df: dataframe
            dataframe with the training dataset
        Returns
        -------
        dataframe
            dataframe with the features scaled.
    """
    train_mean = train_df.mean()
    #train_mean.to_csv("pm_2_5_mean.csv", sep=";")
    train_std = train_df.std()
    #train_std.to_csv("pm_2_5_std.csv", sep=";")

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (data - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    #plt.figure(figsize=(12, 16))
    #ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    #_ = ax.set_xticklabels(data.keys(), rotation=45)
    #plt.show()
    # Return
    return train_df, val_df, test_df

def split_data(data, train_per, val_per, test_per):
    """
    Split the data into training, validation and testing datasets:

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        train_per: float
            Percentage of samples for training
        val_per: float
            Percentage of samples for validation
        test_per: float
            Percentage of samples for testing
        Returns
        -------
        dataframe
            Training data frame
        dataframe
            Validation data frame
        dataframe
            Testing data frame
    """
    # Split data
    n = len(data)
    train_df = data[0:int(n*train_per)]
    val_df = data[int(n*train_per):int(n*(train_per+val_per))]
    test_df = data[int(n*(train_per+val_per)):]
    # Return
    return train_df, val_df, test_df

def prepare_forecasting_dataset(data, target_names=['target'], lags=1, ahead=1, dropnan=True):
    """
    Prepare dataset for forecasting problems:

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        Returns
        -------
        dataframe
            dataframe in forecasting format
    """
    # Prepare forecasting dataset
    Xdata = data.drop(target_names, inplace=False, axis=1)
    Ydata = data[target_names]
    n_vars = 1 if type(Xdata) is list else Xdata.shape[1]
    #df = pd.DataFrame(data)
    cols, names = list(), list()
    columns = Xdata.columns
    # input sequence (t-n, ... t-1)
    for i in range(lags, 0, -1):
        cols.append(Xdata.shift(i))
        names += [(columns[j]+'(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+m)
    for i in range(0, ahead):
        cols.append(Ydata.shift(-i))
        if i == 0:
            names += [(target+'(t)') for target in target_names]
        else:
            names += [(target+'(t+%d)' % (i)) for target in target_names]
    # put it all together
    dataset = pd.concat(cols, axis=1)
    dataset.columns = names
    # drop rows with NaN values
    if dropnan:
        dataset.dropna(inplace=True)
    # Split input/target data
    dataset_X, dataset_Y = dataset.iloc[:,:-ahead*len(target_names)], dataset.iloc[:,-ahead*len(target_names):]
    index = dataset_Y.index
    # Return
    return dataset_X.reset_index(drop=True), dataset_Y.reset_index(drop=True), index

def data_windowing(data, train_per, val_per, test_per, lags, ahead, label_columns):
    """
    Prepare data for forecasting:
        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        train_per: float
            Percentage of samples for training
        val_per: float
            Percentage of samples for validation
        test_per: float
            Percentage of samples for testing
        lags: integer
            Number of time periods from the past to use for training
        ahead: integer
            Number of time periods to predict
        label_columns: list of string
            List of target features
        Returns
        -------
        WindowGenerator
            Object containing the data for training, 
            validation and testing ready for forecasting
    """
    train_df,val_df,test_df = split_data(data,train_per,val_per,test_per)
    train_df,val_df,test_df = scale_data_meanstd(data,train_df,val_df,test_df)

#    train_df.to_csv("input_pm_10.csv",sep=';')

    w = WindowGenerator(input_width=lags, label_width=ahead, shift=ahead,
                        train_df=train_df,val_df=val_df,test_df=test_df,label_columns=label_columns)
    
    # example_window = tf.stack([np.array(train_df[3470:3470+w.total_window_size]),
    #                            np.array(train_df[3500:3500+w.total_window_size]),
    #                            np.array(train_df[4120:4120+w.total_window_size])])

    # example_inputs, example_labels = w.split_window(example_window)

    # print('All shapes are: (batch, time, features)')
    # print(f'Window shape: {example_window.shape}')
    # print(f'Inputs shape: {example_inputs.shape}')
    # print(f'labels shape: {example_labels.shape}')

    # w.plot()

    # print(w.train.element_spec)

    # for example_inputs, example_labels in w.train.take(1):
    #     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    #     print(f'Labels shape (batch, time, features): {example_labels.shape}')

    # Return
    return w

def plot_timeserie(data):
    """
    Plot time serie target variables.

        Parameters
        ----------
        data: dataframe
            dataframe with the data of all turbines
        Returns
        -------
        None
    """
    # Plot time serie
    values = data.values
    columns = data.columns
    i = 1
    plt.figure(figsize=(20, 20))
    for j in range(len(columns)):
        plt.subplot(len(columns), 1, i)
        plt.plot(values[:, j])
        plt.title(columns[j], y=0.5, loc='right')
        i += 1
    plt.show()

def prepare_data(data):
    """
    Prepare the data
        Parameters
        ----------
        data: dataframe
            Raw data
        Returns
        -------
        dataframe
            dataframe ready for training
    """
    #____________________________________________________________________________________ 
    # Check time series gaps
    print("    2.5 Checking GAPS")
    check_timeseries_gaps(data)
    #____________________________________________________________________________________ 
    # Null/Missing values
    print("    Checking missing values")
    data = check_missing_values(data)
    #____________________________________________________________________________________
    # Date time feature extraction
    print("    Date time feature extraction")
    data = datetime_feature_extraction(data)
    #____________________________________________________________________________________ 
    # Remove outliers
    print("    Removing outliers")
    data = remove_outliers(data)
    #print(data.describe().transpose())
    #_ = data.reset_index(drop=True).plot(subplots=True)
    #plt.show()
    #____________________________________________________________________________________ 
    # Correlations
    print("    Correlations")
    data = perform_feature_selection_growing_inputs(data, 'pm_10', 0.05)
    #____________________________________________________________________________________ 
    # Window generator
    print("    Generating windows")
    lags_list = [5,10,20,50]
    w_list = []
    for lag in lags_list:
        w_list.append(data_windowing(data, 0.7, 0.2, 0.1, lag, 5, ["pm_10"]))
    #____________________________________________________________________________________
    # Return
    return w_list, lags_list