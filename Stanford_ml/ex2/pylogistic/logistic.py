#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd


def sigmoid(z):
    """"""
    g = 1 / (1 + np.exp(-z))
    return g


def cost_function(theta, X, y):
    """"""
    m = y.size
    J = (-y.T * np.log(sigmoid(X * theta)) - \
         (1 - y).T * np.log(1 - sigmoid(X * theta))) / m
    grad = ((sigmoid(X * theta) - y).T * X) / m
    return J, grad


def gradient_descent(X, y, theta, alpha):
    """"""
    m = y.size
    row, col = X.shape
    temp_theta = []
    for i in range(col):
        _temp = theta[i] - alpha * ((sigmoid(X * theta) - y).T * X[:, i])
        temp_theta.append(_temp.tolist()[0][0])
    theta = np.array(temp_theta)
    return theta


def train(X, y, theta, alpha, num_iters):
    """"""
    cost_history = range(num_iters)
    for i in range(num_iters):
        _theta = gradient_descent(X, y, theta, alpha)
        theta = np.mat(_theta, dtype=float).T
        cost_history[i] = cost_function(theta, X, y)
    return theta


def performance(testData, testY, theta):
    """"""
    z = testData * theta
    g = sigmoid(z)
    count = 0

    for v in g - testY:
        if v != 0:
            count += 1
    return count / float(testY.size)



def predict():
    """"""
    pass


if __name__ == '__main__':
    data = np.mat(pd.read_csv('train.csv', header=None), dtype=float)
    _x = data[:, range(data.shape[1])[0:-1]]
    X = np.insert(_x, 0, values=1, axis=1)
    y = data[:, -1]

    theta = np.mat(np.zeros(X.shape[1], dtype=float), dtype=float).T
    alpha, num_iters = 0.01, 1000

    theta = train(X, y, theta, alpha, num_iters)
    print(theta)
    # get the performance of Model
    _test_data = np.mat(pd.read_csv('test.csv', header=None), dtype=float)
    test_data = _test_data[:, range(_test_data.shape[1])[0:-1]]
    testData = np.insert(test_data, 0, values=1, axis=1)
    testY = _test_data[:, -1]
    print(performance(testData, testY, theta))

