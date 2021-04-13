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
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, f1_score
import numpy as np
import os

import warnings
from sklearn.exceptions import UndefinedMetricWarning


def check_and_create_xgb_folders(properties):
    if not os.path.exists(f'{properties.output_dir}/xgboost/'):
        os.mkdir(f'{properties.output_dir}/xgboost/')

    if not os.path.exists(f'{properties.output_dir}/xgboost/trees/'):
        os.mkdir(f'{properties.output_dir}/xgboost/trees/')


# noinspection PyAttributeOutsideInit
class GradientForest:
    def __init__(self, properties, antibiotic_name, num_classes, max_depth=5, eta=0.2, early_stopping_rounds=10):
        self.num_classes = num_classes
        self.num_folds = 5
        self.early_stopping_rounds = early_stopping_rounds
        self.antibiotic_name = antibiotic_name

        # A warning is given talking about evaluation metric changing for 'multi:softprob' objective
        # Exact:  Starting in XGBoost 1.3.0, the default evaluation metric used with the objective
        #         'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric
        #         if you'd like to restore the old behavior.
        # To remove it, we set 'verbosity': 0
        self.params = {'max_depth': max_depth, 'eta': eta, 'objective': "multi:softprob", 'num_class': num_classes, 'verbosity': 0}
        self.properties = properties
        self.model = None
        self.feat_importance = None

    def train(self, train_data, train_labels):
        """Create, Train, and export an XGBoost model along with its CV results. Return the model."""

        train_dmatrix = xgb.DMatrix(train_data, label=train_labels)

        cv_results = xgb.cv(
            params=self.params,
            dtrain=train_dmatrix,
            nfold=self.num_folds,
            early_stopping_rounds=self.early_stopping_rounds,
            metrics='mlogloss'
        )

        model = xgb.train(params=self.params, dtrain=train_dmatrix, num_boost_round=len(cv_results))

        cv_results.to_csv(f'{self.properties.output_dir}/cv_results/xgboost_{self.antibiotic_name}.csv')
        model.save_model(f'{self.properties.output_dir}models/xgboost_{self.antibiotic_name}.model')
        model.dump_model(f'{self.properties.output_dir}xgboost/trees/{self.antibiotic_name}.txt')

        self.model = model
        self.feat_importance = pd.DataFrame(list(model.get_fscore().items()),
                                            columns=['feature', 'importance']).sort_values('importance',
                                                                                           ascending=False)

    def test(self, test_data, test_labels):
        if self.model is not None:
            test_dmatrix = xgb.DMatrix(test_data, label=test_labels)
            predictions = self.model.predict(test_dmatrix)
            binary_labels = label_binarize(test_labels, classes=[i for i in range(self.num_classes)])

            with warnings.catch_warnings():
                # This gets thrown when calculating ROC values if there are 0 True Positives for an MIC (which can happen)
                # Exact warning: UndefinedMetricWarning: No positive samples in y_true, true positive value should be meaningless
                warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

                # this will suppress all warnings in this block
                warnings.simplefilter("ignore")

                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(self.num_classes):
                    fpr[i], tpr[i], _ = roc_curve(binary_labels[:, i], predictions[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), predictions.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                y_pred = np.argmax(predictions, axis=1)
                y_true = test_labels.values
                class_list = range(self.num_classes)
                f1 = dict()
                f1_scores = f1_score(y_true, y_pred, average=None, labels=class_list)
                f1_micro = f1_score(y_true, y_pred, average='micro')

            return fpr, tpr, roc_auc, f1_scores, f1_micro
        else:
            raise AttributeError("Model not made yet. Run train function before test function.")

    def get_feature_importance(self):
        if self.feat_importance is not None:
            return self.feat_importance
        else:
            raise AttributeError("Model not made yet. Run train function before getting feature importance values.")

    def save_feature_importance(self):
        if self.feat_importance is not None:
            self.feat_importance.to_csv(f'{self.properties.output_dir}xgboost/feature_importance_{self.antibiotic_name}.csv', index=False)
        else:
            raise AttributeError("Model not made yet. Run train function before saving feature importance values.")