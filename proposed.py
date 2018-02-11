import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys

from scipy.special import logit
from scipy.special import expit as logistic
from sklearn.model_selection import KFold

from common.seasonal   import calc_trend, detrend, calc_seasonal, calc_irregular, decompose, \
    calc_df_irregular, calc_df_trend
from common.timeseries import cross_validate_split, fix_inf, differential
from common.evaluation import corr_coef, mse, mape
from common.printer    import display_top_terms

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


m = 52

def calc_corr_seasonal(X, y):
    # logger.info('Seasonal correlation')
    # The first column of X is dates, we don't use it
    terms = list(X)[1:]
    scores = []

    # y decomposition
    y = fix_inf(logit(y).values)
    y_trend, y_seasonal, y_irregular = decompose(y, m)
    for term in terms:
        # x decomposition
        x = fix_inf(logit(X[term].values))
        x_trend, x_seasonal, x_irregular = decompose(x, m)
        scores.append((term, corr_coef(x_seasonal[:m], y_seasonal[:m])))

    return scores


def calc_corr_trend(X, y):
    # logger.info('Trend correlation')
    # The first column of X is dates, we don't use it
    terms = list(X)[1:]
    scores = []

    # y decomposition
    y = fix_inf(logit(y).values)
    y_trend, y_seasonal, y_irregular = decompose(y, m)
    y_trend_diff = [differential(y_trend, delta)
                    for delta in range(1,4)]
    for term in terms:
        # x decomposition
        x = fix_inf(logit(X[term].values))
        x_trend, x_seasonal, x_irregular = decompose(x, m)
        x_trend_diff = [differential(x_trend, delta)
                        for delta in range(1,4)]
        trend_scores = [corr_coef(_x, _y)
                        for _x, _y
                        in zip(x_trend_diff, y_trend_diff)]
        scores.append((term, np.max(trend_scores)))

    return scores


def calc_corr_irregular(X, y):
    # logger.info('Irregular correlation')
    # The first column of X is dates, we don't use it
    terms = list(X)[1:]
    scores = []

    # y decomposition
    y = fix_inf(logit(y).values)
    y_trend, y_seasonal, y_irregular = decompose(y, m)
    for term in terms:
        # x decomposition
        x = fix_inf(logit(X[term].values))
        x_trend, x_seasonal, x_irregular = decompose(x, m)
        scores.append((term, corr_coef(x_irregular, y_irregular)))

    return scores


def rank(scores, seasonal_scores):
    overall_scores = []
    for (term, score), (_, seasonal_score) \
            in zip(scores, seasonal_scores):
        overall_scores.append((term, score * seasonal_score))
    overall_scores.sort(key=lambda x: x[1], reverse=True)
    return overall_scores


def train(X, y, alpha):
    _X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, _X).fit_regularized(L1_wt=0, alpha=alpha)
    return model


def predict(model, X):
    _X = sm.add_constant(X, has_constant='add')
    y_predict = model.predict(_X)
    return y_predict


def score_mse(X, y, alpha):
    y = pd.Series(y)
    kf = KFold(n_splits=5)
    X_cv = kf.split(X)
    y_cv = kf.split(y)
    mean_mse = 0

    for (X_train_index, X_eval_index), \
        (y_train_index, y_eval_index) in zip(X_cv, y_cv):

        X_train = X.iloc[X_train_index]
        y_train = y.iloc[y_train_index]
        X_eval  = X.iloc[X_eval_index]
        y_eval  = y.iloc[y_eval_index]

        model = train(X_train, y_train, alpha)
        y_predict = predict(model, X_eval)
        mean_mse += mse(y_predict, y_eval) / 5

    return mean_mse


def subset_select(X, y, ranking):
    # alpha is the L2 regularization parameter
    alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]
    best_mse = np.inf
    best_alpha = None
    best_selected = None

    for alpha in alphas:
        tmp_mse = np.inf
        tmp_selected = []
        num_selected = 0
        agg_matrix = pd.DataFrame()
        for term, _ in ranking:
            agg_matrix[num_selected] = X[term]
            mse = score_mse(agg_matrix, y, alpha)
            if mse < tmp_mse:
                failed_count = 0
                tmp_mse = mse
                tmp_selected += [term]
                num_selected += 1
            else:
                failed_count += 1
                if failed_count >= 5:
                    break
        if tmp_mse < best_mse:
            best_mse      = tmp_mse
            best_selected = tmp_selected
            best_alpha    = alpha

    return best_selected, best_alpha


def aggregate(X, subset):
    agg_matrix = pd.DataFrame()
    for idx, term in enumerate(subset):
        agg_matrix[idx] = X[term]
    return agg_matrix


def experiment(disease_no, lag):
    # For seasonal correlation, we always use no-lag data
    dir = 'data/0/'
    X_train = pd.read_csv(dir + 'D{}_X_train.csv'.format(disease_no), index_col=0)
    y_train = pd.read_csv(dir + 'D{}_y_train.csv'.format(disease_no), index_col=0)
    y_train = y_train['infection-rate']
    corr_seasonal = calc_corr_seasonal(X_train, y_train)

    dir = 'data/{}/'.format(lag)
    X_train = pd.read_csv(dir + 'D{}_X_train.csv'.format(disease_no), index_col=0)
    y_train = pd.read_csv(dir + 'D{}_y_train.csv'.format(disease_no), index_col=0)
    X_test  = pd.read_csv(dir + 'D{}_X_test.csv'.format(disease_no), index_col=0)
    y_test  = pd.read_csv(dir + 'D{}_y_test.csv'.format(disease_no), index_col=0)
    y_train = y_train['infection-rate']
    y_test  = y_test['infection-rate']

    # FEATURE SELECTION
    print('- Ranking feature ...')
    corr_trend        = calc_corr_trend(X_train, y_train)
    corr_irregular    = calc_corr_irregular(X_train, y_train)
    ranking_trend     = rank(corr_trend, corr_seasonal)
    ranking_irregular = rank(corr_irregular, corr_seasonal)
    display_top_terms(ranking_trend, 'for TREND')
    display_top_terms(ranking_irregular, 'for IRREGULAR')

    # Calculate components for the data frame
    X_train_trend     = calc_df_trend(X_train.drop('date', axis=1), 52)
    X_train_irregular = calc_df_irregular(X_train.drop('date', axis=1), 52)
    y_train_trend, _, y_train_irregular = decompose(fix_inf(logit(y_train).values), 52)

    print('- Selecting best feature subset ... ', end='')
    subset_trend    , alpha_trend     = subset_select(X_train_trend    , y_train_trend    , ranking_trend)
    subset_irregular, alpha_irregular = subset_select(X_train_irregular, y_train_irregular, ranking_irregular)
    agg_x_train_trend     = aggregate(X_train_trend, subset_trend)
    agg_x_train_irregular = aggregate(X_train_irregular, subset_irregular)
    print('selected', len(subset_trend), 'for trend,', len(subset_irregular), 'for irregular.')
    print("- Selected search terms saved at "
          "'output/selected/T_{}_{}.txt' and "
          "'output/selected/T_{}_{}.txt'."
          .format(disease_no, lag, disease_no, lag))

    # Logging selected term
    with open('output/selected/T_{}_{}.txt'.format(disease_no, lag), 'w') as f:
        for term in subset_trend:
            f.write(term + '\n')

    with open('output/selected/I_{}_{}.txt'.format(disease_no, lag), 'w') as f:
        for term in subset_irregular:
            f.write(term + '\n')

    # RELEARN AND PREDICT
    print('- Learning the final model and predicting...', end=' ')
    model_trend     = train(agg_x_train_trend    , y_train_trend    , alpha_trend)
    model_irregular = train(agg_x_train_irregular, y_train_irregular, alpha_irregular)

    # We will calculate each component invidually for each week in test period
    # We need the train data for decomposing the test time series
    # First, let's make a copy of train data
    X_agg_curr_trend     = aggregate(X_train, subset_trend)
    X_agg_curr_irregular = aggregate(X_train, subset_irregular)
    X_agg_test_trend     = aggregate(X_test , subset_trend)
    X_agg_test_irregular = aggregate(X_test , subset_irregular)

    # We use the seasonal component
    # From the historical data of the epidemic
    _, historical_seasonal, _ = decompose(fix_inf(logit(y_train).values), 52)
    historical_seasonal = list(historical_seasonal)

    predict_y = []
    predict_trends = []
    predict_irregulars = []

    # Now let's predict, one week at a time
    for idx in range(len(X_test.index)):
        # Add data of the new week
        X_agg_curr_trend     = X_agg_curr_trend.append(X_agg_test_trend.loc[idx, :])
        X_agg_curr_irregular = X_agg_curr_irregular.append(X_agg_test_irregular.loc[idx, :])

        # Re-decompose the search time series
        X_curr_trend         = calc_df_trend(X_agg_curr_trend, 52)
        X_curr_irregular     = calc_df_irregular(X_agg_curr_irregular, 52)
        historical_seasonal.append(historical_seasonal[-52])

        # We need only the latest one
        curr_trend     = X_curr_trend.iloc[-1:]
        curr_irregular = X_curr_irregular.iloc[-1:]
        curr_seasonal  = historical_seasonal[-1]

        # Let's predict each component
        predict_trend = predict(model_trend, curr_trend).values[0]
        predict_irregular = predict(model_irregular, curr_irregular).values[0]
        predict_seasonal = curr_seasonal

        # And then add them to the result list
        predict_y.append(logistic(predict_trend * predict_irregular * predict_seasonal))
        predict_trends.append(predict_trend)
        predict_irregulars.append(predict_irregular)

    _mape = mape(y_test, predict_y)
    _coef = corr_coef(y_test, predict_y)

    # THE CODE BELOW IS JUST FOR VISUALIZATION
    predict_y_train_trend     = predict(model_trend, agg_x_train_trend)
    predict_y_train_irregular = predict(model_irregular, agg_x_train_irregular)
    predict_y_train_seasonal  = historical_seasonal[:len(y_train_trend)]
    predict_y_train           = logistic(predict_y_train_trend * predict_y_train_irregular * predict_y_train_seasonal)

    predict_y_all_trend       = np.append(predict_y_train_trend    , np.array(predict_trends))
    predict_y_all_irregular   = np.append(predict_y_train_irregular, np.array(predict_irregulars))
    predict_y_all             = np.append(predict_y_train          , predict_y)

    print('Finished.')
    return _mape, _coef, (predict_y_all_trend,
                          predict_y_all_irregular,
                          predict_y_all)