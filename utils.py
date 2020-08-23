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
models_dir = os.path.join(os.path.dirname(__file__), SETTINGS['MODELS_DIR'])
lgbm_datasets_dir = os.path.join(os.path.dirname(__file__), SETTINGS['LGBM_DATASETS_DIR'])


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


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """ Function that helps scale down memory usage of a data frame
    Args:
        df: Input data frame
        verbose: Boolean value when set to True prints memory usage reduction
    Returns:
        Data Frame with reduced memory usage
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for column in df.columns:
        col_type = df[column].dtypes
        if col_type in numerics:
            c_min = df[column].min()
            c_max = df[column].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[column] = df[column].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[column] = df[column].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[column] = df[column].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[column] = df[column].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[column] = df[column].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[column] = df[column].astype(np.float32)
                else:
                    df[column] = df[column].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
