# src/metrics/eu_regression.py
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .base import Metric


def concordance_cc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    cov = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    return 2 * cov / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


class EmotionRegressionMetrics(Metric):
    def compute(self, y_pred, y_true):
        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "Pearson": pearsonr(y_true, y_pred)[0],
            "CCC": concordance_cc(y_true, y_pred),
        }
