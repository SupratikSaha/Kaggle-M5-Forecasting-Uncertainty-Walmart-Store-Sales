""" Code for training the prediction model """

import gc
import pickle
import tqdm
import warnings
import lightgbm as lgb
from keras_models import keras_model, keras_model_no_en1_en2
from utils import *


def create_model(dense_cols: List[str], end_train: int, x_train: Dict[str, int], y_train: pd.Series,
                 ver: str, embedding: int, model_func: Callable) -> None:
    """ Loads the requested model, trains on training data and saves model weights
    Args:
        dense_cols: List of numerical and boolean columns
        end_train: Last training data day in data
        x_train: Input training features in a dictionary
        y_train: Input training labels
        ver: Model version
        embedding: Number of embeddings to be used
        model_func: Model function used to train the data
    """
    # train
    base_epochs = 30
    model = model_func(num_dense_features=len(dense_cols), lr=0.0001, embedding=embedding)

    model.save_weights(models_dir + 'Keras_CatEmb_final3_et' + str(end_train) + 'ep' +
                       str(base_epochs) + '_ver-' + ver + '.h5')

    for j in range(4):
        model.fit(x_train, y_train,
                  batch_size=2 ** 14,
                  epochs=1,
                  shuffle=True,
                  verbose=2
                  )
        model.save_weights(
            models_dir + 'Keras_CatEmb_final3_et' + str(end_train) + 'ep' +
            str(base_epochs + 1 + j) + '_ver-' + ver + '.h5')

    # del model
    gc.collect()


def train_model() -> None:
    """ Main function to pre-process training data and train using pre built models"""
    warnings.filterwarnings('ignore')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Vars
    seed = 42
    seed_everything(seed)  # We want all results to be deterministic

    # LIMITS and const
    target = 'sales'  # Our target
    p_horizon = 28  # Prediction horizon

    version = '4'  # Our model version
    end_train = 1941

    # PATHS for Features
    original = data_path

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

    # STORES ids
    store_identifiers = pd.read_csv(original + 'sales_train_evaluation.csv')['store_id']
    store_identifiers = list(store_identifiers.unique())

    # SPLITS for lags creation
    rows_split = []
    for i in [1, 7, 14]:
        for j in [7, 14, 30, 60]:
            rows_split.append([i, j])

    grid_df, features_columns = get_data_by_store(store_identifiers[0], target)

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
        grid_df, features_columns = get_data_by_store(store_id, target)

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
    flag = (sales.d < end_train + 1) & (sales.d > end_train + 1 - 17 * 28)
    x_train = make_x(sales[flag], dense_cols, cat_cols)
    x_train['dense1'], scaler1 = preprocess_data(x_train['dense1'])
    y_train = sales["demand"][flag].values

    # First Model
    create_model(dense_cols, end_train, x_train, y_train,
                 ver='With_Emb1', embedding=2, model_func=keras_model)

    # Second Model
    create_model(dense_cols, end_train, x_train, y_train,
                 ver='No_EN1_EN2', embedding=2, model_func=keras_model_no_en1_en2)

    # Third Model
    create_model(dense_cols, end_train, x_train, y_train,
                 ver='Original', embedding=1, model_func=keras_model)
