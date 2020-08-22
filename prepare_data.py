""" Code file that is used to prepare datasets to be used in the predicting models"""

import numpy as np
import pandas as pd
import time
import psutil
import random
import lightgbm as lgb
from math import ceil
from scipy import sparse
from multiprocessing import Pool  # Multiprocess Runs
from typing import Any, Callable, Dict, List
from sklearn.decomposition import PCA
from functools import partial
from utils import data_path, save_data_path, make_normal_lag


def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    """ Get memory size with appropriate units
    Args:
        num: Memory in use
        suffix: Provided suffix - By default B denoting Bytes
    Returns:
        String with memory used formatted to correct units
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


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


def merge_by_concat(df1: pd.DataFrame, df2: pd.DataFrame, merge_on: List[str]) -> pd.DataFrame:
    """ Merge dataframes by concat to not lose dtypes
    Args:
        df1: Left data frame to merge with
        df2: Right data frame to merge
        merge_on: List of column names to merge on
    Return:
        Merged data frame
    """
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [column for column in list(merged_gf) if column not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


def rmse(y: pd.Series, y_pred: pd.Series) -> float:
    """ Function to calculate Root Mean Squared Error
    Args:
        y: Target values
        y_pred: Predicted values
    Returns:
        Calculated RMSE
    """
    return np.sqrt(np.mean(np.square(y - y_pred)))


def make_fast_test(df: pd.DataFrame, remove_features: List[str], lgb_params: Dict[str, Any],
                   end_train: int, target: str) -> lgb:
    """ Function to speed up features tests. estimator = make_fast_test(grid_df).
     Returns lgb booster for future analysis
    Args:
        df: Input data frame
        remove_features: List of columns to be removed
        lgb_params: Parameter dict of LGBM model
        end_train: Integer that indicates last day in training set
        target: Main target column name
    Returns:
        LGBM estimator
    """
    feature_columns = [column for column in list(df) if column not in remove_features]

    tr_x, tr_y = df[df['d'] <= (end_train - 28)][feature_columns], df[df['d'] <= (end_train - 28)][target]
    vl_x, v_y = df[df['d'] > (end_train - 28)][feature_columns], df[df['d'] > (end_train - 28)][target]

    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=v_y)

    estimator = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, valid_data],
        verbose_eval=500,
    )

    return estimator


def make_pca(df: pd.DataFrame, pca_col: str, n_days: int, seed: int, target: str) -> pd.DataFrame:
    """ Function that uses PCA and performs 7->3 dimensionality reduction.
    The main question here - can we have almost same rmse boost with less features
    and less dimensionality?
    PCA is "unsupervised" learning and with shifted target we can be sure that we have no target leakage
    Args:
        df: Input data frame
        pca_col: Row Identifier Column name
        n_days: Number of days
        seed: Our random seed for everything
        target: Main target column name
    Returns:
        Data Frame with 3 columns after performing PCA
    """
    print('PCA:', pca_col, n_days)

    # We don't need any other columns to make pca
    pca_df = df[[pca_col, 'd', target]]

    # If we are doing pca for other series "levels" we need to agg first
    if pca_col != 'id':
        # merge_base = pca_df[[pca_col, 'd']]
        pca_df = pca_df.groupby([pca_col, 'd'])[target].agg(['sum']).reset_index()
        pca_df[target] = pca_df['sum']
        del pca_df['sum']

    # Min/Max scaling
    pca_df[target] = pca_df[target] / pca_df[target].max()

    # Making "lag" in old way (not parallel)
    lag_days = [column for column in range(1, n_days + 1)]
    format_s = '{}_pca_' + pca_col + str(n_days) + '_{}'
    pca_df = pca_df.assign(**{
        format_s.format(column, l): pca_df.groupby([pca_col])[column].transform(lambda x: x.shift(l))
        for l in lag_days
        for column in [target]
    })

    pca_columns = list(pca_df)[3:]
    pca_df[pca_columns] = pca_df[pca_columns].fillna(0)
    pca = PCA(random_state=seed)

    # You can use fit_transform here
    pca.fit(pca_df[pca_columns])
    pca_df[pca_columns] = pca.transform(pca_df[pca_columns])

    print(pca.explained_variance_ratio_)

    # we will keep only 3 most "valuable" columns/dimensions
    persist_columns = pca_columns[:3]
    print('Columns to keep:', persist_columns)

    # If we are doing pca for other series "levels" we need merge back our results to merge_base df
    # and only than return resulted df. I'll skip that step here

    return pca_df[persist_columns]


def df_parallelize_run(func: Callable, term_split: List[int], grid_df, n_cores, target):
    """ Function to parallelize runs using multiprocessing. This function is NOT 'bulletproof',
    be careful and pass only correct types of variables.
    Args:
        func: Function to apply on each split
        term_split: Number of lags days
        grid_df: Grid DataFrame
        n_cores: Number of CPU cores being used
        target: Main target column name
    Returns:
    """
    num_cores = np.min([n_cores, len(term_split)])
    pool = Pool(num_cores)
    f = partial(func, grid_df, target)
    df = pd.concat(pool.map(f, term_split), axis=1)
    pool.close()
    pool.join()
    return df


def find_last_sale(df, n_day, target) -> pd.Series:
    """ Function to get last non zero sale
    Args:
        df: Input data frame
        n_day: Number of days
        target: Main target column name
    Returns:
        Returns a column of last non zero sales
    """
    # Limit initial df
    ls_df = df[['id', 'd', target]]

    # Convert target to binary
    ls_df['non_zero'] = (ls_df[target] > 0).astype(np.int8)

    # Make lags to prevent any leakage
    ls_df['non_zero_lag'] = ls_df.groupby(['id'])['non_zero'].transform(
        lambda x: x.shift(n_day).rolling(2000, 1).sum()).fillna(-1)

    temporary_df = ls_df[['id', 'd', 'non_zero_lag']].drop_duplicates(subset=['id', 'non_zero_lag'])
    temporary_df.columns = ['id', 'd_min', 'non_zero_lag']

    ls_df = ls_df.merge(temporary_df, on=['id', 'non_zero_lag'], how='left')
    ls_df['last_sale'] = ls_df['d'] - ls_df['d_min']

    return ls_df[['last_sale']]


def prepare_datasets() -> None:
    """ Main function to prepare the datasets to be used for model building"""
    # We will need some global VARS for future
    # FE Vars
    target = 'sales'  # Our main target
    end_train = 1941  # 1913 + 28  # Last day in train set
    main_index = ['id', 'd']  # We can identify item by these columns
    n_cores = psutil.cpu_count()  # Available CPU cores
    seed = 42  # Our random seed for everything
    random.seed(seed)  # to make all tests "deterministic"
    np.random.seed(seed)

    # Load Data
    print('Load Main Data')

    # Reading all  data without any limitations and data type modification
    train_df = pd.read_csv(data_path + 'sales_train_evaluation.csv')
    prices_df = pd.read_csv(data_path + 'sell_prices.csv')
    calendar_df = pd.read_csv(data_path + 'calendar.csv')

    # Make Grid
    print('Create Grid')

    # We can transform horizontal representation to vertical "view"
    # Our "index" will be 'id','item_id','dept_id','cat_id','store_id','state_id'
    # and labels are 'd_' columns

    index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    grid_df = pd.melt(train_df,
                      id_vars=index_columns,
                      var_name='d',
                      value_name=target)

    # If we look on train_df we se that we don't have a lot of training rows
    # but each day can provide more train data
    print('Train rows:', len(train_df), len(grid_df))

    # To be able to make predictions we need to add "test set" to our grid
    add_grid = pd.DataFrame()
    for i in range(1, 29):
        temp_df = train_df[index_columns]
        temp_df = temp_df.drop_duplicates()
        temp_df['d'] = 'd_' + str(end_train + i)
        temp_df[target] = np.nan
        add_grid = pd.concat([add_grid, temp_df])

    grid_df = pd.concat([grid_df, add_grid])
    grid_df = grid_df.reset_index(drop=True)

    # Remove some temporary DFs
    del temp_df, add_grid

    # We will not need original train_df anymore and can remove it
    del train_df

    # You don't have to use df = df construction, you can use inplace=True instead.
    # like this - grid_df.reset_index(drop=True, inplace=True)
    # Let's check our memory usage
    print("{:>20}: {:>8}".format('Original grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    # We can free some memory by converting "strings" to categorical
    # It will not affect merging and we will not lose any valuable data
    for col in index_columns:
        grid_df[col] = grid_df[col].astype('category')

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Reduced grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    # Product Release date
    print('Release week')

    # It seems that leading zero values in each train_df item row are not real 0 sales but mean
    # absence for the item in the store we can safe some memory by removing such zeros
    # Prices are set by week so it we will have not very accurate release week
    release_df = prices_df.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
    release_df.columns = ['store_id', 'item_id', 'release']

    # Now we can merge release_df
    grid_df = merge_by_concat(grid_df, release_df, ['store_id', 'item_id'])
    del release_df

    # We want to remove some "zeros" rows
    # from grid_df
    # to do it we need wm_yr_wk column
    # let's merge partly calendar_df to have it
    grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk', 'd']], ['d'])

    # Now we can cutoff some rows
    # and safe memory
    grid_df = grid_df[grid_df['wm_yr_wk'] >= grid_df['release']]
    grid_df = grid_df.reset_index(drop=True)

    # Let's check our memory usage
    print("{:>20}: {:>8}".format('Original grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    # Should we keep release week as one of the features?
    # Only good CV can give the answer. Let's minify the release values.
    # Min transformation will not help here as int16 -> Integer (-32768 to 32767)
    # and our grid_df['release'].max() serves for int16 but we have have an idea how to transform
    # other columns in case we will need it
    grid_df['release'] = grid_df['release'] - grid_df['release'].min()
    grid_df['release'] = grid_df['release'].astype(np.int16)

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Reduced grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    # Save part 1
    print('Save Part 1')

    # We have our BASE grid ready and can save it as pickle file for future use (model training)
    grid_df.to_pickle(save_data_path + 'grid_part_1_eval.pkl')

    print('Size:', grid_df.shape)

    # Prices
    print('Prices')

    # We can do some basic aggregations
    prices_df['price_max'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
    prices_df['price_min'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
    prices_df['price_std'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
    prices_df['price_mean'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')

    # and do price normalization (min/max scaling)
    prices_df['price_norm'] = prices_df['sell_price'] / prices_df['price_max']

    # Some items are can be inflation dependent and some items are very "stable"
    prices_df['price_nunique'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')
    prices_df['item_nunique'] = prices_df.groupby(['store_id', 'sell_price'])['item_id'].transform('nunique')

    # I would like some "rolling" aggregations but would like months and years as "window"
    calendar_prices = calendar_df[['wm_yr_wk', 'month', 'year']]
    calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
    prices_df = prices_df.merge(calendar_prices[['wm_yr_wk', 'month', 'year']], on=['wm_yr_wk'], how='left')
    del calendar_prices

    # Now we can add price "momentum" (some sort of) Shifted by week, by month mean and by year mean
    prices_df['price_momentum'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id'])[
        'sell_price'].transform(lambda x: x.shift(1))
    prices_df['price_momentum_m'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'month'])[
        'sell_price'].transform('mean')
    prices_df['price_momentum_y'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'year'])[
        'sell_price'].transform('mean')

    del prices_df['month'], prices_df['year']

    # Merge prices and save part 2
    print('Merge prices and save part 2')

    # Merge Prices
    original_columns = list(grid_df)
    grid_df = grid_df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    keep_columns = [col for col in list(grid_df) if col not in original_columns]
    grid_df = grid_df[main_index + keep_columns]
    grid_df = reduce_mem_usage(grid_df)

    # Save part 2
    grid_df.to_pickle(save_data_path + 'grid_part_2_eval.pkl')
    print('Size:', grid_df.shape)

    # We don't need prices_df anymore
    del prices_df

    # We can remove new columns or just load part_1
    grid_df = pd.read_pickle(save_data_path + 'grid_part_1_eval.pkl')

    # Merge calendar
    grid_df = grid_df[main_index]

    # Merge calendar partly
    i_columns = ['date',
                 'd',
                 'event_name_1',
                 'event_type_1',
                 'event_name_2',
                 'event_type_2',
                 'snap_CA',
                 'snap_TX',
                 'snap_WI']

    grid_df = grid_df.merge(calendar_df[i_columns], on=['d'], how='left')

    # Minify data - 'snap_' columns we can convert to bool or int8
    i_columns = ['event_name_1',
                 'event_type_1',
                 'event_name_2',
                 'event_type_2',
                 'snap_CA',
                 'snap_TX',
                 'snap_WI']
    for col in i_columns:
        grid_df[col] = grid_df[col].astype('category')

    # Convert to DateTime
    grid_df['date'] = pd.to_datetime(grid_df['date'])

    # Make some features from date
    grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
    grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
    grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
    grid_df['tm_y'] = grid_df['date'].dt.year
    grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
    grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x / 7)).astype(np.int8)

    grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
    grid_df['tm_w_end'] = (grid_df['tm_dw'] >= 5).astype(np.int8)

    # Remove date
    del grid_df['date']

    # Save part 3 (Dates)
    print('Save part 3')

    grid_df.to_pickle(save_data_path + 'grid_part_3_eval.pkl')
    print('Size:', grid_df.shape)

    # We don't need calendar_df anymore
    del calendar_df
    del grid_df

    # Some additional cleaning
    # Part 1 - Convert 'd' to int
    grid_df = pd.read_pickle(save_data_path + 'grid_part_1_eval.pkl')
    grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

    # Remove 'wm_yr_wk' as test values are not in train set
    del grid_df['wm_yr_wk']
    grid_df.to_pickle(save_data_path + 'grid_part_1_eval.pkl')

    del grid_df

    # Summary
    # Now we have 3 sets of features
    grid_df = pd.concat([pd.read_pickle(save_data_path + 'grid_part_1_eval.pkl'),
                         pd.read_pickle(save_data_path + 'grid_part_2_eval.pkl').iloc[:, 2:],
                         pd.read_pickle(save_data_path + 'grid_part_3_eval.pkl').iloc[:, 2:]],
                        axis=1)

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Full Grid', sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    print('Size:', grid_df.shape)

    # 2.5GiB + is is still too big to train our model (on kaggle with its memory limits)
    # and we don't have lag features yet. But what if we can train by state_id or shop_id?
    state_id = 'CA'
    grid_df = grid_df[grid_df['state_id'] == state_id]
    print("{:>20}: {:>8}".format('Full Grid', sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    # Full Grid:   1.2GiB

    store_id = 'CA_1'
    grid_df = grid_df[grid_df['store_id'] == store_id]
    print("{:>20}: {:>8}".format('Full Grid', sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    # Full Grid: 321.2MiB
    # Seems its good enough now.

    # Final list of features
    grid_df.info()

    # LAG FEATURES
    # We will need only train dataset to show lags concept
    train_df = pd.read_csv(data_path + 'sales_train_evaluation.csv')

    # To make all calculations faster we will limit dataset by 'CA' state
    train_df = train_df[train_df['state_id'] == 'CA']

    # Data Representation
    # Let's check our shape
    print('Shape', train_df.shape)

    # Horizontal representation

    # If we feed directly this data to model our label will be values in column 'd_1913'
    # all other columns will be our "features". In lag terminology all d_1->d_1912 columns
    # are our lag features (target values in previous time period)
    # Good thing that we have a lot of features here. Bad thing is that we have just 12196 "train rows"
    # Note: here and after all numbers are limited to 'CA' state

    # Vertical representation
    # On other hand we can think of d_ columns as additional labels and can significantly
    # scale up our training set to 23330948 rows

    # Good thing that our model will have greater input for training
    # Bad thing that we are losing lags that we had in horizontal representation and
    # also new data set consumes much more memory

    index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    train_df = pd.melt(train_df,
                       id_vars=index_columns,
                       var_name='d',
                       value_name=target)

    # Some modification
    train_df['d'] = train_df['d'].apply(lambda x: x[2:]).astype(np.int16)
    i_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for col in i_columns:
        train_df[col] = train_df[col].astype('category')

    # Lags creation
    # We have several "code" solutions here. As our dataset is already sorted by d values
    # we can simply shift() values also we have to keep in mind that
    # we need to aggregate values on 'id' level

    # group and shift in loop
    temp_df = train_df[['id', 'd', target]]

    start_time = time.time()
    for i in range(1, 8):
        print('Shifting:', i)
        temp_df['lag_' + str(i)] = temp_df.groupby(['id'])[target].transform(lambda x: x.shift(i))

    print('%0.2f min: Time for loops' % ((time.time() - start_time) / 60))

    # Or same in "compact" manner
    # lag_days = [col for col in range(1, 8)]
    # temp_df = train_df[['id', 'd', target]]
    #
    # start_time = time.time()
    # temp_df = temp_df.assign(**{
    #     '{}_lag_{}'.format(col, l): temp_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
    #     for l in lag_days
    #     for col in [target]
    # })

    print('%0.2f min: Time for bulk shift' % ((time.time() - start_time) / 60))

    # Rolling lags
    # We restored some day sales values from horizontal representation
    # as lag features but just few of them (last 7 days or less)
    # because of memory limits we can't have many lag features
    # How we can get additional information from other days?

    # Rolling aggregations
    # noinspection PyRedeclaration
    temp_df: pd.DataFrame = train_df[['id', 'd', 'sales']]
    start_time = time.time()

    for i in [14, 30, 60]:
        print('Rolling period:', i)
        temp_df['rolling_mean_' + str(i)] = temp_df.groupby(['id'])[target].transform(
            lambda x: x.shift(1).rolling(i).mean())
        temp_df['rolling_std_' + str(i)] = temp_df.groupby(['id'])[target].transform(
            lambda x: x.shift(1).rolling(i).std())

    # lambda x: x.shift(1)
    # 1 day shift will serve only to predict day 1914 for other days you have to shift PREDICT_DAY-1913
    # Such aggregations will help us to restore at least part of the information for our model
    # and out of 14+30+60->104 columns we can have just 6 with valuable information (hope it is sufficient)
    # you can also aggregate by max/skew/median etc also you can try other rolling periods 180,365 etc
    print('%0.2f min: Time for loop' % ((time.time() - start_time) / 60))

    # The result
    print(temp_df[temp_df['id'] == 'HOBBIES_1_002_CA_1_evaluation'].iloc[:20])

    # Same for NaNs values - it's normal because there is no data for
    # 0*(rolling_period),-1*(rolling_period),-2*(rolling_period)
    # Memory usage. Let's check our memory usage
    print("{:>20}: {:>8}".format('Original rolling df', sizeof_fmt(temp_df.memory_usage(index=True).sum())))

    # Can we decrease it?
    # 1. if our dataset are aligned by index you don't need 'id' 'd' 'sales' columns
    temp_df = temp_df.iloc[:, 3:]
    print("{:>20}: {:>8}".format('Values rolling df', sizeof_fmt(temp_df.memory_usage(index=True).sum())))

    # Can we make it even smaller? Carefully change data type and/or
    # use sparse matrix to minify 0s. Also note that lgbm accepts matrices as input
    # that is good for memory reduction
    temp_matrix = sparse.csr_matrix(temp_df)

    # restore to df
    temp_matrix_restored = pd.DataFrame(temp_matrix.todense())
    restored_cols = ['roll_' + str(i) for i in list(temp_matrix_restored)]
    temp_matrix_restored.columns = restored_cols

    # Remove old objects
    del temp_df, train_df, temp_matrix, temp_matrix_restored

    # Apply on grid_df
    # lets read grid from https://www.kaggle.com/kyakovlev/m5-simple-fe
    # to be sure that our grids are aligned by index
    grid_df = pd.read_pickle(save_data_path + 'grid_part_1_eval.pkl')

    # We need only 'id','d','sales' to make lags and rollings
    grid_df = grid_df[['id', 'd', 'sales']]
    shift_day = 28

    # Lags with 28 day shift
    start_time = time.time()
    print('Create lags')

    lag_days = [col for col in range(shift_day, shift_day + 15)]
    grid_df = grid_df.assign(**{
        '{}_lag_{}'.format(col, l): grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
        for l in lag_days
        for col in [target]
    })

    # Minify lag columns
    for col in list(grid_df):
        if 'lag' in col:
            grid_df[col] = grid_df[col].astype(np.float16)

    print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

    # Rollings with 28 day shift
    start_time = time.time()
    print('Create rolling aggregates')

    for i in [7, 14, 30, 60, 180]:
        print('Rolling period:', i)
        grid_df['rolling_mean_' + str(i)] = grid_df.groupby(['id'])[target].transform(
            lambda x: x.shift(shift_day).rolling(i).mean()).astype(np.float16)
        grid_df['rolling_std_' + str(i)] = grid_df.groupby(['id'])[target].transform(
            lambda x: x.shift(shift_day).rolling(i).std()).astype(np.float16)

    # Rollings with sliding shift
    for d_shift in [1, 7, 14]:
        print('Shifting period:', d_shift)
        for d_window in [7, 14, 30, 60]:
            col_name = 'rolling_mean_tmp_' + str(d_shift) + '_' + str(d_window)
            grid_df[col_name] = grid_df.groupby(['id'])[target].transform(
                lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)

    print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

    # Export
    print('Save lags and rollings')
    grid_df.to_pickle(save_data_path + 'lags_df_' + str(shift_day) + '_eval.pkl')

    # MEAN ENCODING
    grid_df = pd.concat([pd.read_pickle(save_data_path + 'grid_part_1_eval.pkl'),
                         pd.read_pickle(save_data_path + 'grid_part_2_eval.pkl').iloc[:, 2:],
                         pd.read_pickle(save_data_path + 'grid_part_3_eval.pkl').iloc[:, 2:]],
                        axis=1)

    # Sub-sampling to make all calculations faster.Keep only 5% of original ids.
    keep_id = np.array_split(list(grid_df['id'].unique()), 20)[0]
    grid_df = grid_df[grid_df['id'].isin(keep_id)].reset_index(drop=True)
    # Let's "inspect" our grid DataFrame
    grid_df.info()

    # Drop some items from "TEST" set part (1914...)
    grid_df = grid_df[grid_df['d'] <= end_train].reset_index(drop=True)

    # Features that we want to exclude from training
    remove_features = ['id', 'd', target]

    # Our baseline model serves to do fast checks of new features performance. We will use LightGBM for our tests
    lgb_params = {
        'boosting_type': 'gbdt',  # Standard boosting type
        'objective': 'regression',  # Standard loss for RMSE
        'metric': ['rmse'],  # as we will use rmse as metric "proxy"
        'subsample': 0.8,
        'subsample_freq': 1,
        'learning_rate': 0.05,  # 0.5 is "fast enough" for us
        'num_leaves': 2 ** 7 - 1,  # We will need model only for fast check
        'min_data_in_leaf': 2 ** 8 - 1,  # So we want it to train faster even with drop in generalization
        'feature_fraction': 0.8,
        'n_estimators': 5000,  # We don't want to limit training (you can change 5000 to any big enough number)
        'early_stopping_rounds': 30,  # We will stop training almost immediately (if it stops improving)
        'seed': seed,
        'verbose': -1,
    }

    # Make baseline model
    # baseline_model = make_fast_test(grid_df, remove_features, lgb_params, end_train, target)

    # Launch parallel lag creation and "append" to our grid
    lags_split = [col for col in range(1, 1 + 7)]
    grid_df = pd.concat([grid_df, df_parallelize_run(make_normal_lag, lags_split, grid_df, n_cores, target)], axis=1)

    # Make features test
    test_model = make_fast_test(grid_df, remove_features, lgb_params, end_train, target)

    # Permutation importance Test - https://www.kaggle.com/dansbecker/permutation-importance
    # Let's create validation dataset and features
    features_columns = [col for col in list(grid_df) if col not in remove_features]
    validation_df = grid_df[grid_df['d'] > (end_train - 28)].reset_index(drop=True)

    # Make normal prediction with our model and save score
    validation_df['preds'] = test_model.predict(validation_df[features_columns])
    base_score = rmse(validation_df[target], validation_df['preds'])
    print('Standard RMSE', base_score)

    # Now we are looping over all our numerical features
    for col in features_columns:

        # We will make validation set copy to restore features states on each run
        temp_df = validation_df.copy()

        # Error appears here if we have "categorical" features and can't
        # do np.random.permutation without disrupt categories, so we need to check if feature is numerical
        if temp_df[col].dtypes.name != 'category':
            temp_df[col] = np.random.permutation(temp_df[col].values)
            temp_df['preds'] = test_model.predict(temp_df[features_columns])
            cur_score = rmse(temp_df[target], temp_df['preds'])

            # If our current rmse score is less than base score
            # it means that feature most probably is a bad one and our model is learning on noise
            print(col, np.round(cur_score - base_score, 4))

    # Remove Temp data
    del temp_df, validation_df

    # Remove test features as we will compare performance with baseline model for now
    keep_cols = [col for col in list(grid_df) if 'sales_lag_' not in col]
    grid_df = grid_df[keep_cols]

    # Lets test far away Lags (7 days with 56 days shift) and check permutation importance
    lags_split = [col for col in range(56, 56 + 7)]
    grid_df = pd.concat([grid_df, df_parallelize_run(make_normal_lag, lags_split, grid_df, n_cores, target)], axis=1)
    test_model = make_fast_test(grid_df, remove_features, lgb_params, end_train, target)

    features_columns = [col for col in list(grid_df) if col not in remove_features]
    validation_df = grid_df[grid_df['d'] > (end_train - 28)].reset_index(drop=True)
    validation_df['preds'] = test_model.predict(validation_df[features_columns])
    base_score = rmse(validation_df[target], validation_df['preds'])
    print('Standard RMSE', base_score)

    for col in features_columns:
        temp_df = validation_df.copy()
        if temp_df[col].dtypes.name != 'category':
            temp_df[col] = np.random.permutation(temp_df[col].values)
            temp_df['preds'] = test_model.predict(temp_df[features_columns])
            cur_score = rmse(temp_df[target], temp_df['preds'])
            print(col, np.round(cur_score - base_score, 4))

    del temp_df, validation_df

    # Remove test features as we will compare performance with baseline model for now
    keep_cols = [col for col in list(grid_df) if 'sales_lag_' not in col]
    grid_df = grid_df[keep_cols]
    # Make PCA
    grid_df = pd.concat([grid_df, make_pca(grid_df, 'id', 7, seed, target)], axis=1)

    # Remove test features. As we will compare performance with baseline model for now
    keep_cols = [col for col in list(grid_df) if '_pca_' not in col]
    grid_df = grid_df[keep_cols]

    # Mean/std target encoding. We will use these three columns for test (in combination with store_id)
    i_columns = ['item_id', 'cat_id', 'dept_id']

    # We will use simple target encoding by std and mean agg
    for col in i_columns:
        print('Encoding', col)
        temp_df = grid_df[grid_df['d'] <= (1913 - 28)]
        # to be sure we don't have leakage in our validation set

        temp_df = temp_df.groupby([col, 'store_id']).agg({target: ['std', 'mean']})
        joiner = '_' + col + '_encoding_'
        temp_df.columns = [joiner.join(col).strip() for col in temp_df.columns]
        temp_df = temp_df.reset_index()
        grid_df = grid_df.merge(temp_df, on=[col, 'store_id'], how='left')
        del temp_df

    # Remove test features
    # keep_cols = [col for col in list(grid_df) if '_encoding_' not in col]
    # grid_df = grid_df[keep_cols]

    # Bad thing is that for some items we are using past and future values.
    # But we are looking for "categorical" similarity on a "long run". So future here is not a big problem.
    # Find last non zero. Need some "dances" to fit in memory limit with groupers
    # grid_df = pd.concat([grid_df, find_last_sale(grid_df, 1, target)], axis=1)

    # Make features test
    # test_model = make_fast_test(grid_df, remove_features, lgb_params, end_train, target)

    # Apply on grid_df. Lets read grid from https://www.kaggle.com/kyakovlev/m5-simple-fe
    # to be sure that our grids are aligned by index
    grid_df = pd.read_pickle(save_data_path + 'grid_part_1_eval.pkl')
    grid_df[target][grid_df['d'] > (1913 - 28)] = np.nan
    base_cols = list(grid_df)

    i_cols = [
        ['state_id'],
        ['store_id'],
        ['cat_id'],
        ['dept_id'],
        ['state_id', 'cat_id'],
        ['state_id', 'dept_id'],
        ['store_id', 'cat_id'],
        ['store_id', 'dept_id'],
        ['item_id'],
        ['item_id', 'state_id'],
        ['item_id', 'store_id']
    ]

    for col in i_cols:
        print('Encoding', col)
        col_name = '_' + '_'.join(col) + '_'
        grid_df['enc' + col_name + 'mean'] = grid_df.groupby(col)[target].transform('mean').astype(np.float16)
        grid_df['enc' + col_name + 'std'] = grid_df.groupby(col)[target].transform('std').astype(np.float16)

    keep_cols = [col for col in list(grid_df) if col not in base_cols]
    grid_df = grid_df[['id', 'd'] + keep_cols]

    print('Save Mean/Std encoding')
    grid_df.to_pickle(save_data_path + 'mean_encoding_df_eval.pkl')
