import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.special import logit
from scipy.special import expit as logistic

from common.timeseries import cross_validate_split, fix_inf
from common.evaluation import corr_coef, mape

import common.printer as printer

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def aggregate(X, subset):
    # len(X.ix[:,0]) is length of the data
    agg_x = np.zeros(len(X.ix[:,0]))
    for (term, _) in subset:
        agg_x += X[term].values

    return agg_x


def train(x, y):
    _x = sm.add_constant(fix_inf(logit(x)), has_constant='add')
    _y = fix_inf(logit(y))
    model = sm.OLS(_y, _x).fit()

    return model


def predict(model, x):
    _x = sm.add_constant(fix_inf(logit(x)), has_constant='add')

    return logistic(model.predict(_x))


def GFT_scoring(x, y):
    x_cv = cross_validate_split(list(x), 5)
    y_cv = cross_validate_split(list(y), 5)
    score = 0
    for (x_train, x_eval), (y_train, y_eval) in zip(x_cv, y_cv):
        model = train(x_train, y_train)
        y_predict = predict(model, x_eval)
        score += corr_coef(y_eval, y_predict) / 5

    # The score is invalid if it's NaN
    if not np.isnan(score):
        return score
    else:
        return -np.inf


def rank(X, y):
    # logger.info('Ranking')
    # The first column of X is dates, we don't use it
    terms = list(X)[1:]
    scores = []
    for term in terms:
        scores.append((term, GFT_scoring(X[term], y)))
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores


def subset_select(X, y, ranking):
    # logger.info('Subset selection')
    best_score = -np.inf
    selected = None
    agg_x = np.zeros(len(y))

    for pos, (term, _) in enumerate(ranking):
        agg_x += X[term].values
        score = GFT_scoring(agg_x, y)
        if score > best_score:
            best_score = score
            selected = pos

    return ranking[:selected+1]


def experiment(disease_no, lag):
    dir = 'data/{}/'.format(lag)
    X_train = pd.read_csv(dir + 'D{}_X_train.csv'.format(disease_no), index_col=0)
    y_train = pd.read_csv(dir + 'D{}_y_train.csv'.format(disease_no), index_col=0)
    X_test = pd.read_csv(dir + 'D{}_X_test.csv'.format(disease_no), index_col=0)
    y_test = pd.read_csv(dir + 'D{}_y_test.csv'.format(disease_no), index_col=0)

    y_train = y_train['infection-rate']
    y_test = y_test['infection-rate']

    # FEATURE SELECTION
    print('- Ranking feature ...')
    ranking = rank(X_train, y_train)
    printer.display_top_terms(ranking, '')

    print('- Selecting best feature subset ... ', end='')
    best_subset = subset_select(X_train, y_train, ranking)
    agg_x_train = aggregate(X_train, best_subset)
    agg_x_test  = aggregate(X_test, best_subset)
    print('selected', len(best_subset), 'search terms.')
    print("- Selected search terms saved at 'output/selected/gft_{}_{}.txt'".format(disease_no, lag))

    # Logging selected term
    with open('output/selected/gft_{}_{}.txt'.format(disease_no, lag), 'w') as f:
        for term, _ in best_subset:
            f.write(term + '\n')

    # RELEARN AND PREDICT
    print('- Learning the final model and predicting...', end=' ')
    final_model = train(agg_x_train, y_train)
    y_predict   = predict(final_model, agg_x_test)
    _mape = mape(y_test, y_predict)
    _coef = corr_coef(y_test, y_predict)

    # THE CODE BELOW IS JUST FOR VISUALIZATION
    y_predict = predict(final_model, np.append(agg_x_train, agg_x_test))

    print('Finished.')
    return _mape, _coef, y_predict
