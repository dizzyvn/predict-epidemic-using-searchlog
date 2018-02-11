import numpy as np
import pandas as pd

from scipy.special import logit
from common.timeseries import fix_inf

def calc_trend(x, m):
    trend = []
    sum_m = np.sum(x[:m])
    for t in range(m, len(x)):
        sum_m += x[t] - x[t - m]
        trend.append(sum_m / m)

    return trend


def detrend(x, trend):
    # Remove the first year of data
    l = len(trend)
    x = x[-l:]
    detrended = []
    for t in range(l):
        if trend[t] == 0.0:
            return None
        detrended.append(x[t] / trend[t])

    return detrended


def calc_seasonal(detrended, m):
    seasonal = np.zeros(len(detrended))
    for t in range(0, m):
        indexes = list(range(t, len(detrended), m))
        sum_t = 0
        for i in indexes:
            sum_t += detrended[i]
        sum_t /= len(indexes)

        for i in indexes:
            seasonal[i] = sum_t

    return seasonal


def calc_irregular(detrended, seasonal):
    irregular = []
    l = len(detrended)
    for t in range(l):
        irregular.append(detrended[t] / seasonal[t])

    return irregular


def decompose(x, m):
    x_trend = calc_trend(x, m)
    x_detrended = detrend(x[m:], x_trend)
    x_seasonal = calc_seasonal(x_detrended, m)
    x_irregular = calc_irregular(x_detrended, x_seasonal)

    return x_trend, x_seasonal, x_irregular


def calc_df_trend(X, m):
    X_trend = pd.DataFrame()
    for term in list(X):
        x = fix_inf(logit(X[term].values))
        X_trend[term] = calc_trend(x, m)
    return X_trend


def calc_df_irregular(X, m):
    X_irregular = pd.DataFrame()
    for term in list(X):
        x = fix_inf(logit(X[term].values))
        _, _, x_irregular = decompose(x, m)
        X_irregular[term] = x_irregular
    return X_irregular

