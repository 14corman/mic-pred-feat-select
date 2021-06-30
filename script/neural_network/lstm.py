# -*- coding: utf-8 -*-
"""
Created on Mar 29 8:29 PM 2021

@author: Cory Kromer-Edwards

The Neural Network (NN) generated here is based off the NN
built in:

"""# -*- coding: utf-8 -*-
"""
Created on Mar 24 7:13 PM 2021

@author: Cory Kromer-Edwards

The Neural Network (NN) generated here is based off the NN
built in:

"""
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import ml_funcs
import pandas as pd
import numpy as np
import tensorflow as tf

import warnings
from sklearn.exceptions import UndefinedMetricWarning


def check_and_create_lstm_folders(properties):
    if not os.path.exists(f'{properties.output_dir}/lstm/'):
        os.mkdir(f'{properties.output_dir}/lstm/')

    if not os.path.exists(f'{properties.analysis_dir}/lstm/'):
        os.mkdir(f'{properties.analysis_dir}/lstm/')


class LstmNN:
    def __init__(self, properties, antibiotic_name, num_genes, classes, batch_size=256):
        self.num_folds = 5
        self.batch_size = batch_size
        self.classes = list(classes)
        self.num_classes = len(classes)
        self.antibiotic_name = antibiotic_name
        self.properties = properties
        self.num_genes = num_genes
        self.models = []
        self.output_model = None
        self.has_trained = False

    def build_model(self, columns):
        num_classes = self.num_classes

        def func():
            input_layer = tf.keras.Input(shape=(len(columns),), name="input")
            embedding_layer = tf.keras.layers.Embedding(27, 50, input_length=len(columns))
            embedded_input = embedding_layer(input_layer)
            x = tf.keras.layers.LSTM(100, activation="tanh")(embedded_input)
            output = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
            model = tf.keras.Model(input_layer, output)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                          metrics=["categorical_accuracy"])

            return model

        return func

    def train(self, train_data, train_labels, combined_labels):
        """Create, Train, and export an Random Forest model along with its CV results. Return the model."""
        nn_data = None
        error_pd = pd.DataFrame([], columns=['Type', 'Model', 'error', 'epoch'])

        for gene_id, gene in enumerate(["ompk35", "ompk36", "ompk37"]):
            labels = train_labels[gene_id].apply(ml_funcs.scale_labels, classes=self.classes)

            gene_tensor = tf.one_hot(labels.filter(items=list(train_data[gene_id].index), axis=0).values, self.num_classes)

            input_dataset = np.asarray(train_data[gene_id].values)
            print(input_dataset)

            # (number of samples, number of time steps [just 1 here since we are passing in whole sequence], number of genes)
            # input_dataset = input_dataset.reshape((len(input_dataset), 1, len(input_dataset[0])))

            # input_dataset = train_data[gene_id].join(gene_labels, how="inner")
            # X_train, X_val, y_train, y_val = train_test_split(input_dataset, gene_labels, test_size=0.2)
            # train_nn_dataset = ml_funcs.df_to_dataset(train_nn_data, batch_size=self.batch_size)
            # val_nn_dataset = ml_funcs.df_to_dataset(val_nn_data, shuffle=False, batch_size=self.batch_size)

            with warnings.catch_warnings():
                # This gets thrown when running Cross Validation.
                # Exact warning: The least populated class in y has only 1 members, which is less than n_splits=5.
                # There is also another deprecation warning with sklearn using a deprecated predict method with the NN
                warnings.simplefilter(action='ignore')
                # est = KerasClassifier(build_fn=self.build_model(list(train_data[gene_id].columns)), epochs=50, batch_size=self.batch_size)
                # cv_results = cross_val_score(est, train_data[gene_id], gene_labels, cv=5, scoring="f1_micro", verbose=0)
                # cv_df = pd.DataFrame([cv_results], columns=["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"])
                # cv_df.to_csv(f'{self.properties.output_dir}/cv_results/nn_lstm_{self.antibiotic_name}.csv')

            self.models.append(self.build_model(list(train_data[gene_id].columns))())
            history = self.models[gene_id].fit(input_dataset, gene_tensor, epochs=100, batch_size=self.batch_size, validation_split=0.2)
            tf.keras.utils.plot_model(self.models[gene_id], show_shapes=True, rankdir="LR",
                                      to_file=f'{self.properties.output_dir}models/nn_lstm_{gene}_{self.antibiotic_name}.png')
            self.models[gene_id].save(f'{self.properties.output_dir}models/nn_lstm_{self.antibiotic_name}.model')

            # Collect error values to be plot later for validation
            for i, x in enumerate(history.history["categorical_accuracy"]):
                if i == 0:
                    continue

                error_pd = error_pd.append(
                    {'Type': "training error", 'Model': gene, 'error': 1 - x, 'epoch': i},
                    ignore_index=True)

            for i, x in enumerate(history.history["val_categorical_accuracy"]):
                if i == 0:
                    continue

                error_pd = error_pd.append(
                    {'Type': "validation error", 'Model': gene, 'error': 1 - x, 'epoch': i},
                    ignore_index=True)

            # prediction_input = ml_funcs.df_to_dataset(input_dataset, batch_size=self.batch_size)
            predictions = self.models[gene_id].predict(input_dataset)
            predictions_df = pd.DataFrame(predictions, index=train_data[gene_id].index,
                                          columns=[f"{gene}_mic_{self.classes[i]}" for i in range(self.num_classes)])

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
        #     Path(f"{self.properties.output_dir}nn_lstm_nn_data_report.html"))

        train_nn_data, val_nn_data = train_test_split(nn_data, test_size=0.2)

        train_nn_dataset = ml_funcs.df_to_dataset(train_nn_data, batch_size=self.batch_size)
        val_nn_dataset = ml_funcs.df_to_dataset(val_nn_data, shuffle=False, batch_size=self.batch_size)

        self.output_model = ml_funcs.build_output_nn(train_nn_dataset, nn_features, self.num_classes)
        tf.keras.utils.plot_model(self.output_model, show_shapes=True, rankdir="LR",
                                  to_file=f'{self.properties.output_dir}models/nn_lstm_out_nn_dense_{self.antibiotic_name}.png')
        history = self.output_model.fit(train_nn_dataset, epochs=20, validation_data=val_nn_dataset)
        self.output_model.save(
            f'{self.properties.output_dir}models/nn_lstm_out_nn_dense_{self.antibiotic_name}.model')

        # Collect error values to be plot later for validation
        for i, x in enumerate(history.history["categorical_accuracy"]):
            if i == 0:
                continue

            error_pd = error_pd.append(
                {'Type': "training error", 'Model': "output", 'error': 1 - x, 'epoch': i},
                ignore_index=True)

        for i, x in enumerate(history.history["val_categorical_accuracy"]):
            if i == 0:
                continue

            error_pd = error_pd.append(
                {'Type': "validation error", 'Model': "output", 'error': 1 - x, 'epoch': i},
                ignore_index=True)

        self.has_trained = True
        return error_pd

    def test(self, test_data, test_labels):
        if self.has_trained:
            labels = test_labels.apply(ml_funcs.scale_labels, classes=self.classes)
            binary_labels_all = label_binarize(labels, classes=[i for i in range(len(self.classes))])

            nn_test_data = None
            fpr_all = []
            tpr_all = []
            roc_auc_all = []
            f1_scores = []
            f1_micro = []
            for gene_id, gene in enumerate(["ompk35", "ompk36", "ompk37"]):
                gene_labels = labels.filter(items=list(test_data[gene_id].index), axis=0)\
                    .rename('target')\
                    .apply(ml_funcs.one_hot, num_classes=self.num_classes)

                binary_labels = np.asarray([list(i) for i in gene_labels.values])
                test_dataset = test_data[gene_id].join(gene_labels, how="inner")
                test_dataset = ml_funcs.df_to_dataset(test_dataset, batch_size=self.batch_size)
                predictions = self.models[gene_id].predict(test_dataset)

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
                        fpr[self.classes[i]], tpr[self.classes[i]], _ = roc_curve(binary_labels[:, i],
                                                                                  predictions[:, i])
                        roc_auc[self.classes[i]] = auc(fpr[self.classes[i]], tpr[self.classes[i]])

                    # Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), predictions.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                    fpr_all.append(fpr)
                    tpr_all.append(tpr)
                    roc_auc_all.append(roc_auc)

                    y_pred = np.argmax(predictions, axis=1)
                    y_true = np.argmax(binary_labels, axis=1)
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
                fpr[self.classes[i]], tpr[self.classes[i]], _ = roc_curve(binary_labels_all[:, i], predictions[:, i])
                roc_auc[self.classes[i]] = auc(fpr[self.classes[i]], tpr[self.classes[i]])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels_all.ravel(), predictions.ravel())
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