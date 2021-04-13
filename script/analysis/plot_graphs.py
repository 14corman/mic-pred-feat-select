# -*- coding: utf-8 -*-
"""
Created on Apr 12 8:09 PM 2021

@author: Cory Kromer-Edwards

Will have multiple plotting functions held within such as:

ROC curve
Feature importance bar graph

"""

import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np


def plot_all_roc(fpr, tpr, roc_auc, properties, file_name):
    plt.figure()

    # Plot all available labels
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'violet'])
    for i, color in zip(tpr.keys(), colors):
        # A curve will not be shown if TPR contains nans.
        if not np.isnan(tpr[i]).any() and i != 'micro':
            plt.plot(fpr[i], tpr[i], color=color,
                     lw=2, label=f'MIC ID {i} ROC curve (area = {roc_auc[i]:0.2f})')

    # Plot chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="chance")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MIC ID ROC Comparison')
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.savefig(f'{properties.analysis_dir}{file_name}.jpeg', bbox_inches='tight')
    plt.clf()


def plot_average_roc(fpr, tpr, roc_auc, properties, file_name):
    plt.figure()

    # Plot micro average
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # Plot chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="chance")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MIC ID ROC Comparison')
    plt.legend(loc="lower right")
    plt.savefig(f'{properties.analysis_dir}{file_name}.jpeg', bbox_inches='tight')
    plt.clf()


def plot_feat_imp_bar_graph(feat_importance, properties, file_name):
    # Get and sort feature importances
    n_largest = feat_importance.nlargest(15, ['importance'])

    # Get top 15 features
    plt.barh(n_largest['feature'], n_largest['importance'])
    plt.xlabel("Xgboost Feature Importance")
    plt.savefig(f'{properties.analysis_dir}{file_name}.jpeg', bbox_inches='tight')
    plt.clf()
