""" Code file that sets data paths and hosts functions used by other files"""

import os
import json
import numpy as np
import pandas as pd

path = os.path.join(os.path.dirname(__file__), 'SETTINGS.json')
with open(path, 'r') as f:
    datafile = f.read()
SETTINGS = json.loads(datafile)

data_path = os.path.join(os.path.dirname(__file__), SETTINGS['RAW_DATA_DIR'])
save_data_path = os.path.join(os.path.dirname(__file__), SETTINGS['PROCESSED_DATA_DIR'])


def make_normal_lag(grid_df: pd.DataFrame, target: str, lag_day: int) -> pd.DataFrame:
    """Function to make lags creation faster using our normal Lags (7 days)
       Some more info about lags here: https://www.kaggle.com/kyakovlev/m5-lags-features
       Args:
           grid_df: Input data frame
           target: Main target column name
           lag_day: Days of lag
       Returns:
           Shifts target by the number of lag days and returns the data frame
    """
    lag_df = grid_df[['id', 'd', target]]
    column_name = 'sales_lag_' + str(lag_day)
    lag_df[column_name] = lag_df.groupby(['id'])[target].transform(lambda x: x.shift(lag_day)).astype(np.float16)
    return lag_df[[column_name]]
