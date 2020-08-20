import os
import json
import numpy as np

path = os.path.join(os.path.dirname(__file__), 'SETTINGS.json')
with open(path, 'r') as f:
    datafile = f.read()
SETTINGS = json.loads(datafile)

data_path = os.path.join(os.path.dirname(__file__), SETTINGS['RAW_DATA_DIR'])
save_data_path = os.path.join(os.path.dirname(__file__), SETTINGS['PROCESSED_DATA_DIR'])


def make_normal_lag(grid_df, target, lag_day):
    lag_df = grid_df[['id', 'd', target]]  # not good to use df from "global space"
    column_name = 'sales_lag_' + str(lag_day)
    lag_df[column_name] = lag_df.groupby(['id'])[target].transform(lambda x: x.shift(lag_day)).astype(np.float16)
    return lag_df[[column_name]]
