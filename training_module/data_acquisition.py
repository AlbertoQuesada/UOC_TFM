'''
D A T A   A C Q U I S I T I O N

Created on 11/04/2021
@author: Alberto Quesada
'''
### IMPORTS ###
import pandas as pd
import matplotlib.pyplot as plt

#### DEFINES ###
def load_data(filename,colnames):
    """
        Parameters
        ----------
        data_folder: string
            The directory where the data is.
        colnames: list
            Header to be set.
        Returns
        -------
        dataframe
    """
    # Load CSV as dataframe
    data = pd.read_csv(filename,sep=',',header=None,index_col=False,names=colnames)
    # Reset index as timestamp
    data.index = pd.to_datetime(data.pop('timestamp'), format='%Y-%m-%d %H:%M:%S')
    # Filter last N datapoints
    data = data[(data["temperature"] > 0) & (data["pm_2_5"] < 50)]
    # Show data summary
    print(data.describe().transpose())
    # Plot raw data
    _ = data.reset_index(drop=True).plot(subplots=True)
    plt.show()
    # Return
    return data.drop("pm_2_5", axis=1, inplace=False)
