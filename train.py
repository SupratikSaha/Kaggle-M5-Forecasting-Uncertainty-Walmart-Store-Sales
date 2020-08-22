""" Code for training the prediction model"""

import os
import gc
import pickle
import random
import tqdm
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from keras_models import create_model17EN1EN2emb1, create_model17noEN1EN2, create_model17
from utils import data_path, save_data_path, models_dir, lgbm_datasets_dir


# Seeder :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def get_data_by_store(store, lag_features, mean_features, remove_features, target, start_train,
                      base_path, price_path, calendar_path, mean_encoding_path, lags_path):
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


# Recombine Test set after training
def get_base_test(store_identifiers):
    base_test = pd.DataFrame()

    for storeId in store_identifiers:
        temp_df = pd.read_pickle('lgbtrainings/test_' + storeId + '.pkl')
        temp_df['store_id'] = storeId
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)

    return base_test


# KERAS
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df


def prep_calendar(df):
    df = df.drop(["date", "weekday"], axis=1)
    df = df.assign(d=df.d.str[2:].astype(int))
    df = df.fillna("missing")
    cols = list(set(df.columns) - {"wm_yr_wk", "d"})
    df[cols] = OrdinalEncoder(dtype="int").fit_transform(df[cols])
    df = reduce_mem_usage(df)
    return df


def prep_selling_prices(df):
    gr = df.groupby(["store_id", "item_id"])["sell_price"]
    df["sell_price_rel_diff"] = gr.pct_change()
    df["sell_price_roll_sd7"] = gr.transform(lambda x: x.rolling(7).std())
    df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())
    df = reduce_mem_usage(df)

    return df


def reshape_sales(df, drop_d=None):
    if drop_d is not None:
        df = df.drop(["d_" + str(val + 1) for val in range(drop_d)], axis=1)
    df = df.assign(id=df.id.str.replace("_validation", ""))
    df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1941 + val + 1) for val in range(1 * 28)])
    df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                 var_name='d', value_name='demand')
    df = df.assign(d=df.d.str[2:].astype("int16"))

    return df


def prep_sales(df):
    df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    df['rolling_mean_28_7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    df['rolling_mean_28_28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(28).mean())

    df['rolling_median_28_7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).median())
    df['rolling_median_28_28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(28).median())
    df = reduce_mem_usage(df)

    return df


def preprocess_data(x, scaler=None):
    if not scaler:
        scaler = MinMaxScaler((0, 1))
        scaler.fit(x)
    x = scaler.transform(x)

    return x, scaler


# Input dict for training with a dense array and separate inputs for each embedding input
def make_x(df, dense_cols, cat_cols):
    x = {"dense1": df[dense_cols].values}
    for _, value in enumerate(cat_cols):
        x[value] = df[[value]].values

    return x


def create_model(dense_cols, et, x_train, y_train, base_epochs, ver, model_func) -> None:
    # train
    model = model_func(num_dense_features=len(dense_cols), lr=0.0001)

    model.save_weights(models_dir + 'Keras_CatEmb_final3_et' + str(et) + 'ep' +
                       str(base_epochs) + '_ver-' + ver + '.h5')

    for j in range(4):
        model.fit(x_train, y_train,
                  batch_size=2 ** 14,
                  epochs=1,
                  shuffle=True,
                  verbose=2
                  )
        model.save_weights(
            models_dir + 'Keras_CatEmb_final3_et' + str(et) + 'ep' + str(base_epochs + 1 + j) + '_ver-' + ver + '.h5')

    # del model
    gc.collect()


def train_model() -> None:
    warnings.filterwarnings('ignore')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Vars
    seed = 42  # We want all things
    seed_everything(seed)  # to be as deterministic

    # LIMITS and const
    target = 'sales'  # Our target
    p_horizon = 28  # Prediction horizon

    version = '4'  # Our model version
    start_train = 0  # 1186  # We can skip some rows (Nans/faster training)
    end_train = 1941

    # PATHS for Features
    original = data_path  #
    base_path = save_data_path + 'grid_part_1_eval.pkl'
    price_path = save_data_path + 'grid_part_2_eval.pkl'
    calendar_path = save_data_path + 'grid_part_3_eval.pkl'
    lags_path = save_data_path + 'lags_df_28_eval.pkl'
    mean_encoding_path = save_data_path + 'mean_encoding_df_eval.pkl'

    # Helper to load data by store ID
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
        'rolling_mean_tmp_14_30', 'rolling_mean_tmp_14_60'
    ]

    # Model params
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'tweedie',
        'tweedie_variance_power': 1.1,
        'metric': 'rmse',
        'subsample': 0.6,
        'subsample_freq': 1,
        'learning_rate': 0.02,
        'num_leaves': 2 ** 11 - 1,
        'min_data_in_leaf': 2 ** 12 - 1,
        'feature_fraction': 0.6,
        'max_bin': 100,
        'n_estimators': 1600,
        'boost_from_average': False,
        'verbose': -1,
        'num_threads': 12,
        'seed': seed
    }

    # FEATURES to remove
    remove_features = ['id', 'state_id', 'store_id',
                       'date', 'wm_yr_wk', 'd', target]

    mean_features = ['enc_cat_id_mean', 'enc_cat_id_std',
                     'enc_dept_id_mean', 'enc_dept_id_std',
                     'enc_item_id_mean', 'enc_item_id_std']

    # STORES ids
    store_identifiers = pd.read_csv(original + 'sales_train_evaluation.csv')['store_id']
    store_identifiers = list(store_identifiers.unique())

    # SPLITS for lags creation
    rows_split = []
    for i in [1, 7, 14]:
        for j in [7, 14, 30, 60]:
            rows_split.append([i, j])

    grid_df, features_columns = get_data_by_store(store_identifiers[0], lag_features, mean_features,
                                                  remove_features, target, start_train, base_path,
                                                  price_path, calendar_path, mean_encoding_path,
                                                  lags_path)

    rounds_per_store1 = {
        'CA_1': 700,
        'CA_2': 1100,
        'CA_3': 1600,
        'CA_4': 1500,
        'TX_1': 1000,
        'TX_2': 1000,
        'TX_3': 1000,
        'WI_1': 1600,
        'WI_2': 1500,
        'WI_3': 1100
    }

    # Train Models
    for store_id in store_identifiers:
        print('Train', store_id)
        lgb_params['n_estimators'] = rounds_per_store1[store_id]

        # Get grid for current store
        grid_df, features_columns = get_data_by_store(store_id, lag_features, mean_features,
                                                      remove_features, target, start_train, base_path,
                                                      price_path, calendar_path, mean_encoding_path,
                                                      lags_path)

        train_mask = grid_df['d'] <= end_train
        valid_mask = train_mask & (grid_df['d'] > (end_train - p_horizon))  # pseudo validation
        preds_mask = grid_df['d'] > (end_train - 100)

        # Apply masks and save lgb dataset as bin to reduce memory spikes during data type conversions
        # To avoid any conversions, we should always use np.float32 or save to bin before start training
        train_data = lgb.Dataset(grid_df[train_mask][features_columns],
                                 label=grid_df[train_mask][target])
        train_data.save_binary(lgbm_datasets_dir + 'train_data.bin')
        train_data = lgb.Dataset(lgbm_datasets_dir + 'train_data.bin')

        valid_data = lgb.Dataset(grid_df[valid_mask][features_columns],
                                 label=grid_df[valid_mask][target])

        # Saving part of the dataset for later predictions
        # Removing features that we need to calculate recursively
        grid_df = grid_df[preds_mask].reset_index(drop=True)
        keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
        grid_df = grid_df[keep_cols]
        grid_df.to_pickle(lgbm_datasets_dir + 'test_' + store_id + '.pkl')
        del grid_df

        # Launch seeder again to make lgb training 100% deterministic with each "code line"
        # np.random "evolves", so we need (may want) to "reset" it
        seed_everything(seed)
        estimator = lgb.train(lgb_params,
                              train_data,
                              valid_sets=[valid_data],
                              verbose_eval=100,
                              )

        # Save model - it's not real '.bin' but a pickle file
        # estimator = lgb.Booster(model_file='model.txt')
        # Can only predict with the best iteration (or the saving iteration)
        # pickle.dump gives us more flexibility
        # like estimator.predict(TEST, num_iteration=100)
        # num_iteration - number of iteration want to predict with,
        # NULL or <= 0 means use best iteration

        model_name = models_dir + 'lgbm_finalmodel_' + store_id + '_v' + str(version) + '.bin'
        pickle.dump(estimator, open(model_name, 'wb'))

        # Remove temporary files and objects to free some hdd space and ram memory
        os.remove(lgbm_datasets_dir + "train_data.bin")
        del train_data, valid_data, estimator
        gc.collect()

    # "Keep" models features for predictions
    model_features = features_columns

    with open(lgbm_datasets_dir + 'lgbm_features.txt', 'w') as file_handle:
        for list_item in model_features:
            file_handle.write('%s\n' % list_item)

    # Read raw data
    calendar = pd.read_csv(data_path + "calendar.csv")
    selling_prices = pd.read_csv(data_path + "sell_prices.csv")
    sales = pd.read_csv(data_path + "sales_train_evaluation.csv")

    # Prepare data for keras
    calendar = prep_calendar(calendar)
    selling_prices = prep_selling_prices(selling_prices)
    sales = reshape_sales(sales, 1000)
    sales = sales.merge(calendar, how="left", on="d")
    gc.collect()

    sales = sales.merge(selling_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])
    sales.drop(["wm_yr_wk"], axis=1, inplace=True)
    gc.collect()
    sales = prep_sales(sales)

    del selling_prices

    cat_id_cols = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]
    cat_cols = cat_id_cols + ["wday", "month", "year", "event_name_1",
                              "event_type_1", "event_name_2", "event_type_2"]

    # Encode data with loop to minimize memory use
    for i, v in tqdm.tqdm(enumerate(cat_id_cols)):
        sales[v] = OrdinalEncoder(dtype="int").fit_transform(sales[[v]])

    sales = reduce_mem_usage(sales)
    gc.collect()

    # add feature
    sales['logd'] = np.log1p(sales.d - sales.d.min())

    num_cols = ["sell_price",
                "sell_price_rel_diff",
                "rolling_mean_28_7",
                "rolling_mean_28_28",
                "rolling_median_28_7",
                "rolling_median_28_28",
                "logd"]
    bool_cols = ["snap_CA", "snap_TX", "snap_WI"]
    dense_cols = num_cols + bool_cols

    # Need to do column by column due to memory constraints
    for i, v in tqdm.tqdm(enumerate(num_cols)):
        sales[v] = sales[v].fillna(sales[v].median())

    gc.collect()
    base_epochs = 30
    et = 1941
    flag = (sales.d < et + 1) & (sales.d > et + 1 - 17 * 28)
    x_train = make_x(sales[flag], dense_cols, cat_cols)
    x_train['dense1'], scaler1 = preprocess_data(x_train['dense1'])
    y_train = sales["demand"][flag].values

    # First Model
    create_model(dense_cols, et, x_train, y_train,
                 base_epochs, ver='EN1EN2Emb1', model_func=create_model17EN1EN2emb1)

    # Second Model
    create_model(dense_cols, et, x_train, y_train,
                 base_epochs, ver='noEN1EN2', model_func=create_model17noEN1EN2)

    # Third Model
    create_model(dense_cols, et, x_train, y_train,
                 base_epochs, ver='17last', model_func=create_model17)
