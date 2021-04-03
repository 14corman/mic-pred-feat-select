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
from sklearn import metrics


def create_and_train(train_data, train_labels, properties, antibiotic_name,
                     max_depth=5, eta=0.2, early_stopping_rounds=10):
    """Create, Train, and export an XGBoost model along with its CV results. Return the model."""
    num_folds = 5
    objective = "multi:softprob"
    num_classes = 20    # 20 possible MICs (not including NaN which is -1)

    params = {'max_depth': max_depth, 'eta': eta, 'objective': objective, 'num_class': num_classes}
    train_dmatrix = xgb.DMatrix(train_data, label=train_labels)

    cv_results = xgb.cv(
        params=params,
        dtrain=train_dmatrix,
        nfold=num_folds,
        early_stopping_rounds=early_stopping_rounds,
        feval=metrics.f1_score,
        maximize=True
    )

    model = xgb.train(params=params, dtrain=train_dmatrix, num_boost_round=len(cv_results))

    cv_results.to_csv(f'{properties.output_dir}/cv_results/xgboost/{antibiotic_name}.csv')
    model.save_model(f'{properties.output_dir}models/xgboost/{antibiotic_name}.model')
    model.dump_model(f'{properties.output_dir}xgboost_trees/{antibiotic_name}.txt')

    return model


