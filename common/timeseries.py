import math

import numpy as np
np.seterr(invalid='ignore')

def shift_backward(x, c):
    if c == 0:
        return x
    x = np.array(x)
    return x[:-c]


def shift_forward(x, c):
    if c == 0:
        return x
    x = np.array(x)
    return x[c:]


def differential(X, c):
    res = []
    for t in range(c, len(X)):
        res.append(X[t] - X[t-c])
    return res


def cross_validate_split(x, k):
    l = math.floor(len(x) / k)
    splits = []
    for i in range(k):
        left = l * i
        if i == k-1:
            right = len(x)
        else:
            right = left + l
        train = x[:left] + x[right:]
        test = x[left:right]
        splits.append((train, test))

    return splits


def tr_test_split(xs, y):
    pivot = round(len(y) * 0.8)
    y_train = y[:pivot]
    y_test = y[pivot:]
    xs_train = []
    xs_test = []
    for i in range(len(xs)):
        x = xs[i]
        xs_train.append(x[:pivot])
        xs_test.append(x[pivot:])
    return xs_train, y_train, xs_test, y_test


# Fill -inf with the min value
def fix_inf(x):
    min = np.inf
    # Find the not-inf min value
    for idx in range(len(x)):
        if x[idx] != -np.inf and x[idx] < min:
            min = x[idx]
    # Fill -inf with the min value
    for idx in range(len(x)):
        if x[idx] == -np.inf:
            x[idx] = min
    return x