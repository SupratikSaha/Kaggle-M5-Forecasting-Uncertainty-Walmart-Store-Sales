""" Code file that sets data paths and hosts functions used by other files"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


path = os.path.join(os.path.dirname(__file__), 'SETTINGS.json')
with open(path, 'r') as f:
    datafile = f.read()
SETTINGS = json.loads(datafile)

data_path = os.path.join(os.path.dirname(__file__), SETTINGS['RAW_DATA_DIR'])
save_data_path = os.path.join(os.path.dirname(__file__), SETTINGS['PROCESSED_DATA_DIR'])
models_dir = os.path.join(os.path.dirname(__file__), SETTINGS['MODELS_DIR'])
lgbm_datasets_dir = os.path.join(os.path.dirname(__file__), SETTINGS['LGBM_DATASETS_DIR'])
submission_dir = SETTINGS['SUBMISSION_DIR']


def make_normal_lag(grid_df: pd.DataFrame, target: str, lag_day: int) -> pd.DataFrame:
    """Function to facilitate faster lags creation using our normal Lags (7 days)
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


# Helper to make dynamic rolling lags
def make_lag_roll(base_test: pd.DataFrame, target: str, lag_day: List[int]):
    """Function to facilitate faster dynamic rolling lags creation using our normal Lags (7 days)
       Args:
           base_test: Input data frame
           target: Main target column name
           lag_day: Days of lag
       Returns:
           Shifts target by the number of lag days and returns the data frame
    """
    shift_day = lag_day[0]
    roll_wind = lag_day[1]
    lag_df = base_test[['id', 'd', target]]
    col_name = 'rolling_mean_tmp_' + str(shift_day) + '_' + str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[target].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())

    return lag_df[[col_name]]


def seed_everything(seed: int = 0):
    """ Seeder function to make all processes deterministic
    Args:
        seed: seed value
    """
    random.seed(seed)
    np.random.seed(seed)


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


def get_data_by_store(store: int, target: str) -> Tuple[pd.DataFrame, List[str]]:
    """ Helper function to load data by store ID
    Args:
        store: Store Id whose data is to be loaded
        target: Main target column name
    Returns:
        Tuple of data read for the store in DataFrame and a list of features in the DataFrame
    """
    base_path = save_data_path + 'grid_part_1_eval.pkl'
    price_path = save_data_path + 'grid_part_2_eval.pkl'
    calendar_path = save_data_path + 'grid_part_3_eval.pkl'
    mean_encoding_path = save_data_path + 'mean_encoding_df_eval.pkl'
    lags_path = save_data_path + 'lags_df_28_eval.pkl'

    start_train = 0  # We can skip some rows (Nans/faster training)

    # FEATURES to remove
    remove_features = ['id', 'state_id', 'store_id',
                       'date', 'wm_yr_wk', 'd', target]

    mean_features = ['enc_cat_id_mean', 'enc_cat_id_std',
                     'enc_dept_id_mean', 'enc_dept_id_std',
                     'enc_item_id_mean', 'enc_item_id_std']

    # Read data
    lag_features = [
        'sales_lag_28', 'sales_lag_29', 'sales_lag_30', 'sales_lag_31',
        'sales_lag_32', 'sales_lag_33', 'sales_lag_34', 'sales_lag_35',
        'sales_lag_36', 'sales_lag_37', 'sales_lag_38',
        'sales_lag_39', 'sales_lag_40', 'sales_lag_41', 'sales_lag_42',
        'rolling_mean_7', 'rolling_mean_14',
        'rolling_mean_30', 'rolling_std_30', 'rolling_mean_60',
        'rolling_std_60', 'rolling_mean_180', 'rolling_std_180',
        'rolling_mean_tmp_1_7', 'rolling_mean_tmp_1_14',
        'rolling_mean_tmp_1_30', 'rolling_mean_tmp_1_60',
        'rolling_mean_tmp_7_7', 'rolling_mean_tmp_7_14',
        'rolling_mean_tmp_7_30', 'rolling_mean_tmp_7_60',
        'rolling_mean_tmp_14_7', 'rolling_mean_tmp_14_14',
        'rolling_mean_tmp_14_30', 'rolling_mean_tmp_14_60']

    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(base_path),
                    pd.read_pickle(price_path).iloc[:, 2:],
                    pd.read_pickle(calendar_path).iloc[:, 2:]],
                   axis=1)

    # Leave only relevant store
    df = df[df['store_id'] == store]

    df2 = pd.read_pickle(mean_encoding_path)[mean_features]
    df2 = df2[df2.index.isin(df.index)]

    df3 = pd.read_pickle(lags_path)[lag_features]
    df3 = df3[df3.index.isin(df.index)]

    df = pd.concat([df, df2], axis=1)
    del df2  # to not reach memory limit

    df = pd.concat([df, df3], axis=1)
    del df3  # to not reach memory limit

    # Create features list
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id', 'd', target] + features]

    # Skipping first n rows
    df = df[df['d'] >= start_train].reset_index(drop=True)

    return df, features


def prep_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """ Pre process calendar data
    Args:
        df: Calendar data loaded in a DataFrame
    Returns:
        Pre processed calendar data
    """
    df = df.drop(["date", "weekday"], axis=1)
    df = df.assign(d=df.d.str[2:].astype(int))
    df = df.fillna("missing")
    cols = list(set(df.columns) - {"wm_yr_wk", "d"})
    df[cols] = OrdinalEncoder(dtype="int").fit_transform(df[cols])
    df = reduce_mem_usage(df)
    return df


def prep_selling_prices(df: pd.DataFrame) -> pd.DataFrame:
    """ Pre process selling prices data
    Args:
        df: Selling prices data loaded in a DataFrame
    Returns:
        Pre processed selling prices data
    """
    gr = df.groupby(["store_id", "item_id"])["sell_price"]
    df["sell_price_rel_diff"] = gr.pct_change()
    df["sell_price_roll_sd7"] = gr.transform(lambda x: x.rolling(7).std())
    df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())
    df = reduce_mem_usage(df)

    return df


def reshape_sales(df: pd.DataFrame, drop_d: int = None) -> pd.DataFrame:
    """ Function to reshape sales data
    Args:
        df: Sales data loaded in a data frame
        drop_d: Number of sales columns that are to be dropped
    Returns:
        Reshaped sales data
    """
    if drop_d is not None:
        df = df.drop(["d_" + str(val + 1) for val in range(drop_d)], axis=1)
    df = df.assign(id=df.id.str.replace("_validation", ""))
    df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1941 + val + 1) for val in range(1 * 28)])
    df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                 var_name='d', value_name='demand')
    df = df.assign(d=df.d.str[2:].astype("int16"))

    return df


def prep_sales(df: pd.DataFrame) -> pd.DataFrame:
    """ Pre process sales data
    Args:
        df: Sales data loaded in a data frame
    Returns:
        Pre processed sales data
    """
    df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    df['rolling_mean_28_7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    df['rolling_mean_28_28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(28).mean())

    df['rolling_median_28_7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).median())
    df['rolling_median_28_28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(28).median())
    df = reduce_mem_usage(df)

    return df


def preprocess_data(x: int, scaler: Callable = None) -> Tuple[int, Callable]:
    """ Function to pre process numerical input data by scaling.
    Uses MinMaxScaler by default
    Args:
        x: Input numerical features of training data
        scaler: Scaler function used to pre process numerical input data
    Returns:
        Returns tuple of Scaled numerical input data and the scaler used
    """
    if not scaler:
        scaler = MinMaxScaler((0, 1))
        scaler.fit(x)
    x = scaler.transform(x)

    return x, scaler


def make_x(df: pd.DataFrame, dense_cols: List[str], cat_cols: List[str]) -> Dict[str, int]:
    """ Create an input dictionary for training as a dense array
    and separate inputs for each embedding input
    Args:
        df: Input training data in a data frame
        dense_cols: List of numerical and boolean columns
        cat_cols: List of categorical columns
    Returns:
         Returns training data as a dictionary
    """
    x = {"dense1": df[dense_cols].values}
    for _, value in enumerate(cat_cols):
        x[value] = df[[value]].values

    return x
