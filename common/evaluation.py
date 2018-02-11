import numpy as np

from sklearn.metrics import mean_squared_error

def mape(true, pred):
    true, pred = np.array(true), np.array(pred)
    return np.mean(np.abs(true - pred) / (np.abs(true))) * 100


def mse(true, pred):
    return mean_squared_error(true, pred)


def corr_coef(true, pred):
    return np.corrcoef(true, pred)[0][1]
