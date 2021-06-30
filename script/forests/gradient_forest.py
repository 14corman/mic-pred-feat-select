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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import ml_funcs
import numpy as np
import os
import tensorflow as tf

import warnings
from sklearn.exceptions import UndefinedMetricWarning


def check_and_create_xgb_folders(properties):
    if not os.path.exists(f'{properties.output_dir}/xgboost/'):
        os.mkdir(f'{properties.output_dir}/xgboost/')

    if not os.path.exists(f'{properties.analysis_dir}/xgboost/'):
        os.mkdir(f'{properties.analysis_dir}/xgboost/')

    if not os.path.exists(f'{properties.output_dir}/xgboost/trees/'):
        os.mkdir(f'{properties.output_dir}/xgboost/trees/')


class GradientForest:
    def __init__(self, properties, antibiotic_name, classes, max_depth=50, early_stopping_rounds=50, batch_size=256):
        self.batch_size = batch_size
        self.num_classes = len(classes)
        self.num_folds = 5
        self.early_stopping_rounds = early_stopping_rounds
        self.antibiotic_name = antibiotic_name

        # A warning is given talking about evaluation metric changing for 'multi:stric used with the objective
        #         #         'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric
        #         #         if you'd like to restore the old behavior.oftprob' objective
        # Exact:  Starting in XGBoost 1.3.0, the default evaluation me
        # To remove it, we set 'verbosity': 0
        self.params = {'max_depth': max_depth, 'learning_rate': 0.01, 'objective': "multi:softprob", 'num_class': self.num_classes, 'verbosity': 0}
        self.properties = properties
        self.models = []
        self.feat_importances = []
        self.output_model = None
        self.classes = list(classes)

    def train(self, train_data, train_labels, combined_labels):
        """Create, Train, and export an XGBoost model along with its CV results. Return the model."""
        nn_data = None

        for gene_id, gene in enumerate(["ompk35", "ompk36", "ompk37"]):
            labels = train_labels[gene_id].apply(ml_funcs.scale_labels, classes=self.classes)
            gene_labels = labels.filter(items=list(train_data[gene_id].index), axis=0)
            train_dmatrix = xgb.DMatrix(train_data[gene_id], label=gene_labels)

            cv_results = xgb.cv(
                params=self.params,
                dtrain=train_dmatrix,
                nfold=self.num_folds,
                early_stopping_rounds=self.early_stopping_rounds,
                metrics='mlogloss',
                feval=ml_funcs.f1_eval
            )

            model = xgb.train(params=self.params, dtrain=train_dmatrix, num_boost_round=len(cv_results))

            # Test and save the model
            cv_results.to_csv(f'{self.properties.output_dir}/cv_results/xgboost_{gene}_{self.antibiotic_name}.csv')
            model.save_model(f'{self.properties.output_dir}models/xgboost_{gene}_{self.antibiotic_name}.model')
            model.dump_model(f'{self.properties.output_dir}xgboost/trees/{gene}_{self.antibiotic_name}.txt')

            self.models.append(model)
            self.feat_importances.append(pd.DataFrame(list(model.get_fscore().items()),
                                                columns=['Gene', 'Importance']).sort_values('Importance',
                                                                                               ascending=False))

            predictions = model.predict(train_dmatrix)
            predictions_df = pd.DataFrame(predictions, index=train_data[gene_id].index, columns=[f"{gene}_mic_{self.classes[i]}" for i in range(self.num_classes)])

            if nn_data is None:
                nn_data = predictions_df
            else:
                nn_data = nn_data.join(predictions_df, how="outer")

        # Set up, compile, and fit the output NN
        nn_labels = combined_labels.rename('target').apply(ml_funcs.scale_labels, classes=self.classes)
        nn_labels = nn_labels.apply(ml_funcs.one_hot, num_classes=self.num_classes)
        nn_features = list(nn_data.columns)
        nn_data = nn_data.join(nn_labels, how="inner")
        nn_data = nn_data.fillna(value=-1)

        # from pandas_profiling import ProfileReport
        # from pathlib import Path
        # ProfileReport(nn_data, title="Xgboost NN data report", minimal=True).to_file(
        #     Path(f"{self.properties.output_dir}xgboost_nn_data_report.html"))

        train_nn_data, val_nn_data = train_test_split(nn_data, test_size=0.2)

        train_nn_dataset = ml_funcs.df_to_dataset(train_nn_data, batch_size=self.batch_size)
        val_nn_dataset = ml_funcs.df_to_dataset(val_nn_data, shuffle=False, batch_size=self.batch_size)

        self.output_model = ml_funcs.build_output_nn(train_nn_dataset, nn_features, self.num_classes)
        tf.keras.utils.plot_model(self.output_model, show_shapes=True, rankdir="LR",
                          to_file=f'{self.properties.output_dir}models/nn_dense_out_xgboost_{self.antibiotic_name}.png')
        self.output_model.fit(train_nn_dataset, epochs=100, validation_data=val_nn_dataset)
        self.output_model.save(f'{self.properties.output_dir}models/nn_dense_out_xgboost_{self.antibiotic_name}.model')

    def grid_search(self, train_data, train_labels):
        """
        Peform Grid Search for XGBoost.

        Sources: https://www.kaggle.com/phunter/xgboost-with-gridsearchcv
                 https://www.kaggle.com/vinhnguyen/accelerating-xgboost-with-gpu
        """
        labels = train_labels.apply(ml_funcs.scale_labels, classes=self.classes)
        gs_model = xgb.XGBClassifier(use_label_encoder=False)

        parameters = {'tree_method': ['gpu_hist'],
                      'predictor': ['gpu_predictor'],
                      'objective': ["multi:softprob"],
                      'learning_rate': [0.3, 0.1, 0.05],  # so called `eta` value
                      'max_depth': [1, 6],
                      'eval_metric': ['mlogloss'],
                      'seed': [1337]}

        clf = GridSearchCV(gs_model, parameters, n_jobs=3, cv=10, scoring='f1_micro', verbose=0)
        clf.fit(train_data, labels)
        results = pd.DataFrame(clf.cv_results_)
        results.to_csv(f'{self.properties.output_dir}grid_search/xgboost_run2_{self.antibiotic_name}.csv')
        return results

    def test(self, test_data, test_labels):
        if len(self.models) != 0:
            labels = test_labels.apply(ml_funcs.scale_labels, classes=self.classes)
            binary_labels = label_binarize(labels, classes=[i for i in range(len(self.classes))])

            nn_test_data = None
            fpr_all = []
            tpr_all = []
            roc_auc_all = []
            f1_scores = []
            f1_micro = []
            for gene_id, gene in enumerate(["ompk35", "ompk36", "ompk37"]):
                gene_labels = labels.filter(items=list(test_data[gene_id].index), axis=0)
                binary_gene_labels = label_binarize(gene_labels, classes=[i for i in range(len(self.classes))])
                test_dmatrix = xgb.DMatrix(test_data[gene_id], label=gene_labels)
                predictions = self.models[gene_id].predict(test_dmatrix)

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

                    # Store the FPR, TPR, and AUROC for each MIC index (class)
                    for i in range(len(self.classes)):
                        fpr[self.classes[i]], tpr[self.classes[i]], _ = roc_curve(binary_gene_labels[:, i], predictions[:, i])
                        roc_auc[self.classes[i]] = auc(fpr[self.classes[i]], tpr[self.classes[i]])

                    # Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(binary_gene_labels.ravel(), predictions.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                    fpr_all.append(fpr)
                    tpr_all.append(tpr)
                    roc_auc_all.append(roc_auc)

                    y_pred = np.argmax(predictions, axis=1)
                    y_true = gene_labels.values
                    f1_scores.append(f1_score(y_true, y_pred, average=None, labels=self.classes))
                    f1_micro.append(f1_score(y_true, y_pred, average='micro'))

                    predictions_df = pd.DataFrame(predictions, index=test_data[gene_id].index,
                                                  columns=[f"{gene}_mic_{self.classes[i]}"
                                                           for i in range(self.num_classes)])

                    if nn_test_data is None:
                        nn_test_data = predictions_df
                    else:
                        nn_test_data = nn_test_data.join(predictions_df, how="outer")

            nn_labels = labels.rename('target')
            nn_labels = nn_labels.apply(ml_funcs.one_hot, num_classes=self.num_classes)
            nn_test_data = nn_test_data.join(nn_labels, how="inner")
            nn_test_data = nn_test_data.fillna(value=-1)
            nn_test_data = ml_funcs.df_to_dataset(nn_test_data, shuffle=False, batch_size=self.batch_size)
            predictions = self.output_model.predict(nn_test_data)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            # Store the FPR, TPR, and AUROC for each MIC index (class)
            for i in range(len(self.classes)):
                fpr[self.classes[i]], tpr[self.classes[i]], _ = roc_curve(binary_labels[:, i], predictions[:, i])
                roc_auc[self.classes[i]] = auc(fpr[self.classes[i]], tpr[self.classes[i]])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), predictions.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            fpr_all.append(fpr)
            tpr_all.append(tpr)
            roc_auc_all.append(roc_auc)

            y_pred = np.argmax(predictions, axis=1)
            y_true = test_labels.values
            f1_scores.append(f1_score(y_true, y_pred, average=None, labels=self.classes))
            f1_micro.append(f1_score(y_true, y_pred, average='micro'))

            return fpr_all, tpr_all, roc_auc_all, f1_scores, f1_micro
        else:
            raise AttributeError("Model not made yet. Run train function before test function.")

    def get_feature_importance(self):
        if len(self.feat_importances) != 0:
            return self.feat_importances
        else:
            raise AttributeError("Model not made yet. Run train function before getting feature importance values.")

    def save_feature_importance(self):
        if len(self.feat_importances) != 0:
            for fi, gene in zip(self.feat_importances, ["ompk35", "ompk36", "ompk37"]):
                fi.to_csv(f'{self.properties.output_dir}xgboost/feature_importance_{gene}_{self.antibiotic_name}.csv', index=False)
        else:
            raise AttributeError("Model not made yet. Run train function before saving feature importance values.")
