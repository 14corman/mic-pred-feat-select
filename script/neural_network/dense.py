# -*- coding: utf-8 -*-
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
import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import UndefinedMetricWarning


def check_and_create_dense_folders(properties):
    if not os.path.exists(f'{properties.output_dir}/dense/'):
        os.mkdir(f'{properties.output_dir}/dense/')

    if not os.path.exists(f'{properties.analysis_dir}/dense/'):
        os.mkdir(f'{properties.analysis_dir}/dense/')


def _scale_labels(x, classes=[]):
    """Scaling down labels to be [0, num_classes)"""
    return classes.index(x)  # np.where(classes == x)[0][0]


class DenseNN:
    def __init__(self, properties, antibiotic_name, num_genes, classes):
        self.num_folds = 5
        self.classes = list(classes)
        self.num_genes = num_genes
        self.antibiotic_name = antibiotic_name
        self.properties = properties
        self.model = self.build_model()
        self.has_trained = False

    def build_model(self):
        model = keras.Sequential(
            [
                layers.Dense(2000, activation="relu", input_shape=(self.num_genes,)),
                layers.Dropout(0.5),
                layers.Dense(1000, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(500, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(100, activation="relu"),
                layers.Dense(len(self.classes), activation="softmax", name="output"),
            ])
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_crossentropy', 'sparse_categorical_accuracy'])
        return model

    def train(self, train_data, train_labels):
        """Create, Train, and export an Random Forest model along with its CV results. Return the model."""
        y = train_labels.apply(_scale_labels, classes=self.classes)
        with warnings.catch_warnings():
            # This gets thrown when running Cross Validation.
            # Exact warning: The least populated class in y has only 1 members, which is less than n_splits=5.
            # There is also another deprecation warning with sklearn using a deprecated predict method with the NN
            warnings.simplefilter(action='ignore')
            est = KerasClassifier(build_fn=self.build_model, epochs=50, batch_size=100)
            cv_results = cross_val_score(est, train_data, y, cv=5, scoring="f1_micro", verbose=0)
            cv_df = pd.DataFrame([cv_results], columns=["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"])
            cv_df.to_csv(f'{self.properties.output_dir}/cv_results/nn_dense_{self.antibiotic_name}.csv')

        self.model.fit(train_data, y, batch_size=100, epochs=300)
        self.model.save(f'{self.properties.output_dir}models/nn_dense_{self.antibiotic_name}.model')
        self.has_trained = True

    def test(self, test_data, test_labels):
        if self.has_trained:
            labels = test_labels.apply(_scale_labels, classes=self.classes)
            predictions = self.model.predict(test_data)
            binary_labels = label_binarize(labels, classes=[i for i in range(len(self.classes))])

            with warnings.catch_warnings():
                # This gets thrown when calculating ROC values if there are 0 True Positives for an MIC (which can happen)
                # Exact warning: UndefinedMetricWarning: No positive samples in y_true, true positive value should be meaningless
                warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()

                # Store the FPR, TPR, and AUROC for each MIC index (class)
                for i in range(len(self.classes)):
                    print(binary_labels[:, i])
                    print(predictions[:, i])
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