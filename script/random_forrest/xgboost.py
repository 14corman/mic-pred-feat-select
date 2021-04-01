# -*- coding: utf-8 -*-
"""
Created on Mar 24 6:56 PM 2021

@author: Cory Kromer-Edwards

This file holds the class to generate and work with Gradient
Boosted Decision Trees (using xgboost).

Link for more help with XGBoost library: https://xgboost.readthedocs.io/en/latest/python/python_intro.html

Objective information can be found: https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
"""
import xgboost as xgb

def create_and_train(train_data, labels, max_depth=5, eta=0.2):
    nfolds = 5
    objective = "multi:softmax"
    num_classes =
