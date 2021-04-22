# -*- coding: utf-8 -*-
"""
Created on Mar 24 6:56 PM 2021

@author: Cory Kromer-Edwards

This file holds the class to generate and work with Random Forests

"""
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import os

import warnings
from sklearn.exceptions import UndefinedMetricWarning


def check_and_create_rf_folders(properties):
    if not os.path.exists(f'{properties.output_dir}/rf/'):
        os.mkdir(f'{properties.output_dir}/rf/')

    if not os.path.exists(f'{properties.analysis_dir}/rf/'):
        os.mkdir(f'{properties.analysis_dir}/rf/')


# noinspection PyAttributeOutsideInit
class RandomForest:
    def __init__(self, properties, antibiotic_name):
        self.num_folds = 5
        self.antibiotic_name = antibiotic_name
        self.properties = properties
        self.model = None
        self.feat_importance = None

    def train(self, train_data, train_labels):
        """Create, Train, and export an Random Forest model along with its CV results. Return the model."""

        # For a RF, it is not overall how many classes there are, but what classses it trains on. There
        # may be some missing MICs if they are not in the training dataset.
        classes = train_labels.unique()
        classes.sort()
        self.classes = classes

        model = RandomForestClassifier(min_samples_leaf=10)
        with warnings.catch_warnings():
            # This gets thrown when running Cross Validation.
            # Exact warning: The least populated class in y has only 1 members, which is less than n_splits=5.
            warnings.simplefilter(action='ignore', category=UserWarning)
            cv_results = cross_val_score(model, train_data, train_labels, cv=5, scoring="f1_micro")
            cv_df = pd.DataFrame([cv_results], columns=["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"])

            model.fit(train_data, train_labels)

        cv_df.to_csv(f'{self.properties.output_dir}/cv_results/rf{self.antibiotic_name}.csv')
        joblib.dump(model, f'{self.properties.output_dir}models/rf_{self.antibiotic_name}.joblib')

        self.model = model

        items = []
        for g, i in zip(train_data.columns, model.feature_importances_):
            items.append([g, i])

        self.feat_importance = pd.DataFrame(items,
                                  columns=['Gene', 'Importance']).sort_values('Importance', ascending=False)

    def grid_search(self, train_data, train_labels):
        """
        Peform Grid Search for Random Forest.
        """

        model = RandomForestClassifier()
        parameters = {'max_depth': [1, 15, None],
                      'min_samples_leaf': [1, 10],
                      'min_samples_split': [2, 15],
                      'random_state': [1337]}

        clf = GridSearchCV(model, parameters, n_jobs=3, cv=10, scoring='f1_micro', verbose=0)
        clf.fit(train_data, train_labels)
        results = pd.DataFrame(clf.cv_results_).fillna(-1)  # For None max_depth
        results.to_csv(f'{self.properties.output_dir}grid_search/rf_run1_{self.antibiotic_name}.csv')
        return results

    def test(self, test_data, test_labels):
        if self.model is not None:
            predictions = self.model.predict_proba(test_data)
            binary_labels = label_binarize(test_labels, classes=[i for i in self.classes])

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
                for i in range(len(self.classes)):
                    fpr[self.classes[i]], tpr[self.classes[i]], _ = roc_curve(binary_labels[:, i], predictions[:, i])
                    roc_auc[self.classes[i]] = auc(fpr[self.classes[i]], tpr[self.classes[i]])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), predictions.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                y_pred = np.argmax(predictions, axis=1)
                y_true = test_labels.values
                f1_scores = f1_score(y_true, y_pred, average=None, labels=self.classes)
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
            self.feat_importance.to_csv(f'{self.properties.output_dir}rf/feature_importance_{self.antibiotic_name}.csv', index=False)
        else:
            raise AttributeError("Model not made yet. Run train function before saving feature importance values.")