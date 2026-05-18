import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pr, pv = np.nan, np.nan
    else:
        pr, pv = pearsonr(y_true, y_pred)
        pr, pv = float(pr), float(pv)
    return rmse, mae, r2, pr, pv
