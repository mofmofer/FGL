"""
パラメータフィッティングのグリッドサーチの検索範囲を指定する関数です.
"""

import numpy as np


def param_knn():
    param = {
        'n_neighbors': np.arange(2, 11, 1),
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': np.arange(20, 50, 10),
    }
    return param


def param_rbf(model):
    param = {
        'C': np.arange(9, 20, 2),
        'kernel': [model],
        'degree': [1, 2, 3],
        'gamma': np.arange(0.01, 0.1, 0.01),
        'class_weight': [None, 'balanced']
    }
    return param


def param_linear(model):
    param = {
        'C': [2, 3, 4, 5, 10, 13, 15],
        'kernel': [model],
        'degree': [1, 2, 3],
        'gamma': np.arange(0.01, 0.1, 0.01),
        'class_weight': [None, 'balanced']
    }
    return param


def param_poly(model):
    param = {
        'C': [2, 3, 4, 5, 10, 13, 15],
        'kernel': [model],
        'degree': [1, 2, 3],
        'gamma': np.arange(0.2, 1.6, 0.1),
        'class_weight': [None, 'balanced']
    }
    return param


def param_gbr():
    # param = {'min_samples_split': np.arange(2, 3, 1),'min_samples_leaf': np.arange(10, 20, 10),'max_depth': np.arange(3, 4, 1),'subsample': np.arange(0.6, 0.7, 0.1),'max_features': ['sqrt']}
    param = {'min_samples_split': np.arange(4, 8, 1),'min_samples_leaf': np.arange(80, 150, 10),'max_depth': np.arange(1, 6, 1),'subsample': np.arange(0.5, 0.9, 0.1),'max_features': ['sqrt']}
    return param


def param_rf():
    param = {
        "max_depth": [2, 3, None],
        "n_estimators": [50, 100, 200, 300, 400, 500],
        "max_features": [1, 3, 10],
        "min_samples_split": [2, 3, 10],
        "min_samples_leaf": [1, 3, 10],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }
    return param
