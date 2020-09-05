""" Code file to generate model predictions """

import gc
import time
import pickle
import psutil
import tqdm
from models import keras_model, keras_model_no_en1_en2
from pre_process import df_parallelize_run
from utils import *


def predict_model(dense_cols: List[str], cat_cols: List[str], end_train: int, sales: pd.DataFrame,
                  scaler1: Callable, ver: str, embedding: int, model_func: Callable) -> pd.DataFrame:
    """ Loads the requested model, trains on training data and saves model weights
    Args:
        dense_cols: List of numerical and boolean columns
        cat_cols: List of categorical columns
        end_train: Last training data day in data
        sales: Input training data set
        scaler1: Scaler function used to pre process numerical input data
        ver: Model version
        embedding: Number of embeddings to be used
        model_func: Model function used to train the data
    """
    base_epochs = 30
    val_long = sales[(sales.d >= end_train + 1)].copy()

    model = model_func(num_dense_features=len(dense_cols), lr=0.0001, embedding=embedding)
    for i in range(end_train + 1, end_train + 1 + 28):
        forecast = make_x(val_long[val_long.d == i], dense_cols, cat_cols)
        forecast['dense1'], scaler = preprocess_data(forecast['dense1'], scaler1)

        model.load_weights(
            models_dir + 'Keras_CatEmb_final3_et' + str(end_train) + 'ep' + str(base_epochs) + '_ver-' + ver + '.h5')
        pred = model.predict(forecast, batch_size=2 ** 14)
        for j in range(1, 5):
            model.load_weights(
                models_dir + 'Keras_CatEmb_final3_et' + str(end_train) + 'ep' +
                str(base_epochs + j) + '_ver-' + ver + '.h5')
            pred += model.predict(forecast, batch_size=2 ** 14)
        pred /= 5

        val_long.loc[val_long.d == i, "demand"] = pred.clip(0)  # * 1.02

    val_preds = val_long.demand[(val_long.d >= end_train + 1) & (val_long.d < end_train + 1 + 28)]
    val_preds = val_preds.values.reshape(30490, 28, order='F')

    # write to disk
    ss = pd.read_csv(data_path + 'sample_submission_accuracy.csv')[['id']]
    val_preds_df = pd.DataFrame(val_preds, columns=['F' + str(i) for i in range(1, 29)])
    val_preds_df = pd.concat([ss.id, pd.concat(
        [pd.DataFrame(np.zeros((30490, 28), dtype=np.float32), columns=['F' + str(i) for i in range(1, 29)]),
         val_preds_df], 0).reset_index(drop=True)], 1)
    val_preds_df.to_csv(submission_dir + 'Keras_CatEmb_final3_et' +
                        str(end_train) + '_ver-' + ver + '.csv.gz', index=False, compression='gzip')

    return val_preds_df


def predict_results() -> None:
    """ Main function to generate model predictions """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    seed = 42
    seed_everything(seed)  # We want all results to be deterministic
    n_cores = psutil.cpu_count()  # Available CPU cores

    target = 'sales'  # Our target

    # Store IDs
    store_ids = pd.read_csv(data_path + 'sales_train_evaluation.csv')['store_id']
    store_ids = list(store_ids.unique())

    # Splits for lag creation
    rows_split = []
    for i in [1, 7, 14]:
        for j in [7, 14, 30, 60]:
            rows_split.append([i, j])

    ver = '4'
    end_train = 1941
    model_features = []

    # Open file and read the content in a list
    with open(lgbm_datasets_dir + 'lgbm_features.txt', 'r') as file_handle:
        for line in file_handle:
            # remove linebreak which is the last character of the string
            current_place = line[:-1]
            # add item to the list
            model_features.append(current_place)

    # Predict - Create Dummy DataFrame to store predictions
    all_preds = pd.DataFrame()

    for store_id in store_ids:
        print("store_id: ", store_id)

        # Timer to measure predictions time
        main_time = time.time()

        # Make temporary grid to calculate rolling lags
        grid_df = pd.read_pickle(lgbm_datasets_dir + 'test_' + str(store_id) + '.pkl')
        grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, rows_split,
                                                         grid_df, n_cores, target)], axis=1)

        # Read all our models and make predictions
        model_path = models_dir + 'lgbm_finalmodel_' + store_id + '_v' + str(ver) + '.bin'
        estimator = pickle.load(open(model_path, 'rb'))

        store_preds = pd.DataFrame()

        # Loop over each prediction day. As rolling lags are the most time consuming
        # we will calculate it for whole day
        for predict_day in range(1, 29):
            print('Predict | Day:', predict_day)
            start_time = time.time()

            temp_df = grid_df[grid_df['d'] == (end_train + predict_day)]
            temp_df[target] = estimator.predict(temp_df[model_features])

            # Make good column naming and add to all_preds DataFrame
            temp_df = temp_df[['id', target]]
            temp_df.columns = ['id', 'F' + str(predict_day)]

            if 'id' in list(store_preds):
                store_preds = store_preds.merge(temp_df, on=['id'], how='left')
            else:
                store_preds = temp_df.copy()

            print('#' * 10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F' + str(predict_day)].sum()))
            del temp_df
        if 'id' in list(all_preds):
            all_preds = all_preds.append(store_preds)
        else:
            all_preds = store_preds.copy()

        print('#' * 10, store_id, ' %0.2f min total |' % ((time.time() - main_time) / 60))
        del store_preds

        # Get the actual validation values and concatenate wih evaluation predictions
        val_df = grid_df[(grid_df['d'] >= 1914) & (grid_df['d'] <= 1941)]
        del grid_df

        val_df['d'] = val_df['d'] - 1913
        val_df = val_df.pivot_table(index='id', columns='d', values=target)
        val_df_cols = val_df.columns.to_list()
        for i in range(28):
            val_df_cols[i] = 'F' + str(val_df_cols[i])

        val_df.columns = val_df_cols
        val_df = val_df.reset_index()
        val_df['id'] = val_df['id'].str.replace('evaluation', 'validation')
        all_preds = all_preds.append(val_df)
        del val_df

    all_preds = all_preds.reset_index(drop=True)
    # SAVE TO DISK
    result_df = pd.read_csv(data_path + 'sample_submission_accuracy.csv')[['id']]
    result_df = result_df.merge(all_preds, on=['id'], how='left').fillna(0)
    result_df.to_csv(submission_dir + 'lgbm_final_VER' + str(ver) + '.csv.gz',
                     index=False, compression='gzip')

    del all_preds, result_df
    gc.collect()

    # read raw data
    calendar = pd.read_csv(data_path + "calendar.csv")
    selling_prices = pd.read_csv(data_path + "sell_prices.csv")
    sales = pd.read_csv(data_path + "sales_train_evaluation.csv")

    # prepare data for keras
    calendar = prep_calendar(calendar)
    selling_prices = prep_selling_prices(selling_prices)
    sales = reshape_sales(sales, 1000)

    sales = sales.merge(calendar, how="left", on="d")
    del calendar
    gc.collect()

    sales = sales.merge(selling_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])
    sales.drop(["wm_yr_wk"], axis=1, inplace=True)
    gc.collect()

    sales = prep_sales(sales)

    del selling_prices

    cat_id_cols = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]
    cat_cols = cat_id_cols + ["wday", "month", "year", "event_name_1",
                              "event_type_1", "event_name_2", "event_type_2"]

    # In loop to minimize memory use
    for i, v in tqdm.tqdm(enumerate(cat_id_cols)):
        sales[v] = OrdinalEncoder(dtype="int").fit_transform(sales[[v]])

    sales = reduce_mem_usage(sales)
    gc.collect()

    # add feature
    sales['logd'] = np.log1p(sales.d - sales.d.min())

    # numerical cols
    num_cols = ["sell_price",
                "sell_price_rel_diff",
                "rolling_mean_28_7",
                "rolling_mean_28_28",
                "rolling_median_28_7",
                "rolling_median_28_28",
                "logd"]
    bool_cols = ["snap_CA", "snap_TX", "snap_WI"]
    dense_cols = num_cols + bool_cols

    # Need to set column by column due to memory constraints
    for i, v in tqdm.tqdm(enumerate(num_cols)):
        sales[v] = sales[v].fillna(sales[v].median())

    sales.to_csv(submission_dir + 'Keras_sales_data' + str(end_train) +
                 '_ver-' + ver + '.csv.gz', index=False, compression='gzip')

    # Make train data to use num cols for scaling test data
    flag = (sales.d < end_train + 1) & (sales.d > end_train + 1 - 17 * 28)
    x_train = make_x(sales[flag], dense_cols, cat_cols)
    x_train['dense1'], scaler1 = preprocess_data(x_train['dense1'])
    del x_train

    # predict model 1
    val_preds_df_1 = predict_model(dense_cols, cat_cols, end_train, sales, scaler1,
                                   ver='With_Emb1', embedding=2, model_func=keras_model)
    gc.collect()

    # predict model 2
    val_preds_df_2 = predict_model(dense_cols, cat_cols, end_train, sales, scaler1,
                                   ver='No_EN1_EN2', embedding=2, model_func=keras_model_no_en1_en2)
    gc.collect()

    # predict model 3
    val_preds_df_3 = predict_model(dense_cols, cat_cols, end_train, sales, scaler1,
                                   ver='Original', embedding=1, model_func=keras_model)
    gc.collect()

    ver = 'avg3'
    val_preds_df = val_preds_df_1.copy()
    val_preds_df.iloc[:, 1:] = (val_preds_df_1.iloc[:, 1:].values +
                                val_preds_df_2.iloc[:, 1:].values +
                                val_preds_df_3.iloc[:, 1:].values) / 3
    val_preds_df.to_csv(submission_dir + 'Keras_CatEmb_final3_et' + str(end_train) +
                        '_ver-' + ver + '.csv.gz', index=False, compression='gzip')

    # Ensemble
    # Submissions for M5 accuracy competition, used as starting point to M5 uncertainty
    val_preds_df_lgbm = pd.read_csv(submission_dir + 'lgbm_final_VER4.csv.gz',
                                    compression='gzip')

    final_preds_acc = val_preds_df_lgbm.copy()
    final_preds_acc.iloc[:, 1:] = (val_preds_df_lgbm.iloc[:, 1:] ** 3 * val_preds_df.iloc[:, 1:]) ** (1 / 4)

    sales = pd.read_csv(data_path + "sales_train_evaluation.csv")
    period = 28 * 26
    m = np.mean(sales.iloc[:, -period:].values, 0).mean()
    s = np.mean(sales.iloc[:, -period:].values, 0).std()
    mkd = np.mean(val_preds_df.iloc[30490:, 1:], 0)
    keras_outlier_days = np.where(mkd > m + 6 * s)[0]

    # For days keras preds fails, replace with lgbm preds
    for d in keras_outlier_days:
        final_preds_acc.iloc[:, d + 1] = val_preds_df_lgbm.iloc[:, d + 1].values

    # Take out the evaluation pieces in the prediction
    final_preds_acc = final_preds_acc[final_preds_acc['id'].str.contains('evaluation')]

    # Get the actual validation values
    val_pred_valid = pd.read_csv(data_path + "sales_train_evaluation.csv")
    val_pred_valid_id = val_pred_valid[['id']]
    val_pred_valid_id['id'] = val_pred_valid_id['id'].str.replace('evaluation', 'validation')
    val_pred_valid_sales = val_pred_valid.iloc[:, -28:]
    val_pred_valid_sales.columns = ['F' + str(int(i[2:]) - 1913) for i in list(val_pred_valid_sales.columns)]
    val_pred = pd.concat([val_pred_valid_id, val_pred_valid_sales], axis=1, join='inner')
    final_preds_acc = final_preds_acc.append(val_pred)

    final_preds_acc.to_csv(submission_dir + 'lgbm3keras1.csv.gz', index=False, compression='gzip')
