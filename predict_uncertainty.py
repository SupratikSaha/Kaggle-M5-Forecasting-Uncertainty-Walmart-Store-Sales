""" Code file to predict uncertainties and generate final submissions """

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Any, Dict, List
from utils import data_path, submission_dir


def get_ratios(qs: np.ndarray, coef: float = 0.15) -> np.ndarray:
    """ Generate ratios using normal cumulative distribution function
    Args:
        qs: quantiles with confidence probabilities
        coef: Coefficient value
    Returns:
        Generated ratios using normal cumulative distribution function
    """
    qs2 = np.log(qs / (1 - qs)) * coef
    ratios = stats.norm.cdf(qs2)
    ratios /= ratios[4]
    ratios[-1] *= 1.03
    ratios = pd.Series(ratios, index=qs)

    return ratios.round(3)


def get_ratios2(qs: np.ndarray, coef: float = 0.15, a: float = 1.2) -> np.ndarray:
    """ Generate ratios using skewed normal cumulative distribution function
    Args:
        qs: quantiles with confidence probabilities
        coef: Coefficient value
        a: skewness parameter
    Returns:
        Generated ratios using skewed normal cumulative distribution function
    """
    qs2 = np.log(qs / (1 - qs)) * coef
    ratios = stats.skewnorm.cdf(qs2, a)
    ratios /= ratios[4]
    ratios[-1] *= 1.02
    ratios = pd.Series(ratios, index=qs)

    return ratios.round(3)


def get_ratios3(qs: np.ndarray, coef: float = 0.15, c: float = 0.5, s: float = 0.1) -> np.ndarray:
    """ Generate ratios using power log-normal cumulative distribution function
    Args:
        qs: quantiles with confidence probabilities
        coef: Coefficient value
        c: power parameter
        s: shape parameter
    Returns:
        Generated ratios using power log-normal cumulative distribution function
    """
    qs2 = qs * coef
    ratios = stats.powerlognorm.ppf(qs2, c, s)
    ratios /= ratios[4]
    ratios[0] *= 0.25
    ratios[-1] *= 1.02
    ratios = pd.Series(ratios, index=qs)

    return ratios.round(3)


def widen(array:  np.ndarray, pc: float) -> np.ndarray:
    """ Function to widen the ratio distribution
    Args:
        array : array of ratios
        pc: per cent (0:100)
    Returns:
        Widened array of ratios
    """
    array[array < 1] = array[array < 1] * (1 - pc / 100)
    array[array > 1] = array[array > 1] * (1 + pc / 100)

    return array


def quantile_coefs(q: np.ndarray, level: Any, level_coef_dict: Dict) -> np.ndarray:
    """ Function to get quantiles of a specified level coefficient
    Args:
        q: Repeated array of quantiles with confidence probabilities
        level: Level coefficient(s)
        level_coef_dict: Dictionary of level coefficients
    Returns:
        Returns quantiles of a specified level coefficient
    """
    ratios = level_coef_dict[level]

    return ratios.loc[q].values


def get_group_preds(qs: np.ndarray, pred: pd.DataFrame, level: str,
                    level_coef_dict: Dict, cols: List[str]) -> pd.DataFrame:
    """ Function to get aggregated predictions for a coefficient level
    Args:
        qs: quantiles with confidence probabilities
        pred: Model predictions
        level: Level coefficient
        level_coef_dict: Dictionary of level coefficients
        cols: Column names of aggregated uncertainties in the output file
    Returns:
        Returns aggregated predictions for a coefficient level
    """
    df = pred.groupby(level)[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df] * 9, axis=0, sort=False)
    df.reset_index(inplace=True)
    df[cols] *= np.repeat((quantile_coefs(q, level, level_coef_dict)[:, None]), len(cols), -1)
    if level != "id":
        df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    else:
        df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    df = df[["id"] + list(cols)]

    return df


def get_couple_group_preds(qs: np.ndarray, pred: pd.DataFrame, level1: str, level2: str,
                           level_coef_dict: Dict, cols: List[str]) -> pd.DataFrame:
    """ Function to get aggregated predictions for the couple coefficients
    Args:
        qs: quantiles with confidence probabilities
        pred: Model predictions
        level1: Level coefficient 1
        level2: Level coefficient 2
        level_coef_dict: Dictionary of level coefficients
        cols: Column names of aggregated uncertainties in the output file
    Returns:
        Returns aggregated predictions for the couple coefficients
    """
    df = pred.groupby([level1, level2])[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df] * 9, axis=0, sort=False)
    df.reset_index(inplace=True)
    df[cols] *= np.repeat((quantile_coefs(q, (level1, level2), level_coef_dict)[:, None]), len(cols), -1)
    df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1, lev2, q in
                zip(df[level1].values, df[level2].values, q)]
    df = df[["id"] + list(cols)]
    
    return df


def predict_uncertainties() -> None:
    """ Main function to aggregate predictions and create uncertainty predictions for final submission"""
    acc_filename = 'lgbm3keras1.csv.gz'
    print('creating uncertainty predictions for: ' + acc_filename)
    
    best = pd.read_csv(submission_dir + acc_filename)
    best.iloc[:30490, 1:] = best.iloc[30490:, 1:].values
    
    # Exponential weighted Mean
    c = 0.04
    best.iloc[:, 1:] = best.iloc[:, 1:].ewm(com=c, axis=1).mean().values
    
    # read sales raw data
    sales = pd.read_csv(data_path + "sales_train_evaluation.csv")
    sales = sales.assign(id=sales.id.str.replace("_evaluation", "_validation"))
    
    sub = best.merge(sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on="id")
    sub["_all_"] = "Total"
    
    # Different ratios for different aggregation levels
    # The higher the aggregation level, the more confident we are in the point 
    # prediction --> lower coef, relatively smaller range of quantiles
    qs = np.array([0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995])

    level_coef_dict = {
        "id": widen((get_ratios2(qs, coef=0.3) + get_ratios3(qs, coef=.3, c=0.04, s=0.9)) / 2, pc=0.5),
        "item_id": widen(get_ratios2(qs, coef=0.18, a=0.4), pc=0.5),
        "dept_id": widen(get_ratios(qs, coef=0.04), 0.5),
        "cat_id": widen(get_ratios(qs, coef=0.03), 0.5),
        "store_id": widen(get_ratios(qs, coef=0.035), 0.5),
        "state_id": widen(get_ratios(qs, coef=0.03), 0.5),
        "_all_": widen(get_ratios(qs, coef=0.025), 0.5),
        ("state_id", "item_id"): widen(get_ratios2(qs, coef=0.21, a=0.75), pc=0.5),
        ("state_id", "dept_id"): widen(get_ratios(qs, coef=0.05), 0.5),
        ("store_id", "dept_id"): widen(get_ratios(qs, coef=0.07), 0.5),
        ("state_id", "cat_id"): widen(get_ratios(qs, coef=0.04), 0.5),
        ("store_id", "cat_id"): widen(get_ratios(qs, coef=0.055), 0.5)
    }

    levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
    couples = [("state_id", "item_id"), ("state_id", "dept_id"), ("store_id", "dept_id"),
               ("state_id", "cat_id"), ("store_id", "cat_id")]
    cols = [f"F{i}" for i in range(1, 29)]

    # Make predictions
    df = []
    for level in levels:
        df.append(get_group_preds(qs, sub, level, level_coef_dict, cols))
    for level1, level2 in couples:
        df.append(get_couple_group_preds(qs, sub, level1, level2, level_coef_dict, cols))
    df = pd.concat(df, axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, df], axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    df.loc[df.index >= len(df.index) // 2, "id"] = df.loc[df.index >= len(df.index) // 2, "id"].str.replace(
        "_validation$", "_evaluation")

    # Statistical computation to help overwrite Level12
    sales.sort_values('id', inplace=True)
    sales.reset_index(drop=True, inplace=True)

    quantity_sales = np.average(np.stack((
        sales.iloc[:, -364:].quantile(np.array([0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T,
        sales.iloc[:, -28:].quantile(np.array([0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T,
    )), axis=0, weights=[1, 1.75])

    quantity_sales_w = []

    for i in range(7):
        quantity_sales_w.append(
            np.expand_dims(
                np.average(
                    np.stack(
                        ((sales.iloc[:, np.arange(-28 * 13 + i, 0, 7)].quantile(np.array(
                            [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T).values.reshape(
                            quantity_sales.shape[0] * quantity_sales.shape[1], order='F'),
                         (sales.iloc[:, np.arange(-28 * 3 + i, 0, 7)].quantile(np.array(
                             [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]), axis=1).T).values.reshape(
                             quantity_sales.shape[0] * quantity_sales.shape[1], order='F')), -1), axis=-1), -1)
        )
    quantity_sales_w = np.hstack(quantity_sales_w)

    quantity_sales_w = np.tile(quantity_sales_w, 4)

    medians = np.where(np.array([float(x.split('_')[-2]) for x in df.iloc[:274410, 0]]) == 0.5)[0]
    not_median = np.array([x for x in np.arange(274410) if x not in medians])

    # Overwrite Level12
    df.iloc[not_median, 1:] = (0.2 * df.iloc[not_median, 1:] + 0.7 * np.repeat(
        np.expand_dims(
            quantity_sales.reshape(
                quantity_sales.shape[0] * quantity_sales.shape[1], order='F'), -1), 28, 1)[not_median, :]
                + 0.1 * quantity_sales_w[not_median, :])

    df.iloc[medians, 1:] = (0.8 * df.iloc[medians, 1:] + 0.2 * np.repeat(
        np.expand_dims(
            quantity_sales.reshape(
                quantity_sales.shape[0] * quantity_sales.shape[1], order='F'), -1), 28, 1)[medians, :])

    # Statistical computation to help overwrite Level of 'state_id','item_id' group
    quantity_sales_df = pd.DataFrame(quantity_sales)
    quantity_sales_df['item_id'] = sales['item_id'].values
    quantity_sales_df['state_id'] = sales['state_id'].values
    quantity_sales_df_gb = quantity_sales_df.groupby(['state_id', 'item_id'], as_index=False).mean().iloc[:, 2:]

    sales_g_item_q = quantity_sales_df_gb.values.reshape(quantity_sales_df_gb.shape[0] * quantity_sales_df_gb.shape[1],
                                                         order='F')
    sales_g_item_q = np.repeat(np.expand_dims(sales_g_item_q, -1), 28, 1)

    medians = np.where(np.array([float(x.split('_')[-2]) for x in df.iloc[302067:302067 + 3049 * 3 * 9, 0]]) == 0.5)[0]
    not_median = np.array([x for x in np.arange(302067, 302067 + 3049 * 3 * 9) if x not in medians + 302067])
    
    # Overwrite Level's predictions
    df.iloc[not_median, 1:] = 0.91 * df.iloc[not_median, 1:] + 0.09 * sales_g_item_q[not_median - 302067, :]
    
    # copy preds from first have to second half, predictions for public LB are not right
    df.iloc[int(df.shape[0] / 2):, 1:] = df.iloc[:int(df.shape[0] / 2), 1:].values
    
    # Create final submission file
    ss = pd.read_csv(data_path + 'sample_submission.csv')
    df.columns = ss.columns
    ss = ss[['id']]
    submission = ss.merge(df, on=['id'], how='left')
    submission.to_csv(submission_dir + "submission_uncertainty.csv", index=False)
