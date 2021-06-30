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
import seaborn as sns
import pandas as pd

sns.set_context("paper")


def _free_plot_memory():
    """This goes through and deletes all axes and figures open so that there is no memory leak"""
    plt.clf()
    plt.cla()
    plt.close('all')


def plot_all_roc(fpr, tpr, roc_auc, properties, file_name, mic_set_list):
    plt.figure()

    # Plot all available labels
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'violet'])
    for i, color in zip(tpr.keys(), colors):
        # A curve will not be shown if TPR contains nans.
        if not np.isnan(tpr[i]).any() and i != 'micro':
            plt.plot(fpr[i], tpr[i], color=color,
                     lw=2, label=f'{mic_set_list[i]} ROC curve (area = {roc_auc[i]:0.2f})')

    # Plot chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="chance")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MIC ROC Comparison')
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.savefig(f'{properties.analysis_dir}{file_name}.jpeg', bbox_inches='tight')
    _free_plot_memory()


def plot_single_average_roc(fpr, tpr, roc_auc, properties, file_name):
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
    plt.title('Antibiotic MIC Average ROC Comparison')
    plt.legend(loc="lower right")
    plt.savefig(f'{properties.analysis_dir}{file_name}.jpeg', bbox_inches='tight')
    _free_plot_memory()


def plot_nn_train_error_history(loss_pd, properties, file_name):
    sns.relplot(data=loss_pd, x='epoch', y='error', hue='Type',  col='Model', kind='line')

    sns.despine()  # Remove the top and right graph lines
    plt.savefig(f'{properties.analysis_dir}{file_name}.jpeg', bbox_inches='tight')
    _free_plot_memory()


def plot_all_average_roc(roc, properties, file_name):
    #sns.lineplot(data=roc, x='False Positive Rate', y='True Positive Rate', hue='Antibiotic', style='Algorithm')

    dashes = dict()
    for a in roc['Antibiotic'].unique():
        dashes[a] = ""

    dashes['chance'] = (4, 4)

    for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        roc = roc.append(
            {'Antibiotic': 'chance', 'False Positive Rate': i, 'True Positive Rate': i, 'Algorithm': 'Xgboost'},
            ignore_index=True)
        roc = roc.append(
            {'Antibiotic': 'chance', 'False Positive Rate': i, 'True Positive Rate': i, 'Algorithm': 'Random Forest'},
            ignore_index=True)
        roc = roc.append(
            {'Antibiotic': 'chance', 'False Positive Rate': i, 'True Positive Rate': i, 'Algorithm': 'Dense NN'},
            ignore_index=True)

    #sns.lineplot(data=pd.DataFrame(chance, columns={'False Positive Rate', 'True Positive Rate'}), x='False Positive Rate', y='True Positive Rate')
    #plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    sns.relplot(data=roc, x='False Positive Rate', y='True Positive Rate', hue='Antibiotic',  style='Antibiotic', col='Algorithm', kind='line',
                dashes=dashes)

    sns.despine()  # Remove the top and right graph lines
    plt.savefig(f'{properties.analysis_dir}{file_name}.jpeg', bbox_inches='tight')
    _free_plot_memory()


def plot_feat_imp_bar_graph(feat_importance, properties, file_name):
    # Get and sort feature importances
    n_largest = feat_importance.nlargest(15, ['Importance'])

    # Get top 15 features
    plt.barh(n_largest['Gene'], n_largest['Importance'])
    plt.xlabel("Feature (Gene) Importance")
    plt.savefig(f'{properties.analysis_dir}{file_name}.jpeg', bbox_inches='tight')
    _free_plot_memory()


def plot_gs_results(results, properties):
    """Plot Grid Search results"""
    for var in ['min_samples_leaf', 'min_samples_split', 'max_depth']:
        sns.stripplot(x='Antibiotic', y='F1-micro Score', data=results, jitter=True, hue=var)
        sns.despine()  # Remove the top and right graph lines
        plt.savefig(f'{properties.output_dir}grid_search/rf_{var}_plot.jpeg', bbox_inches='tight')
        _free_plot_memory()


def plot_f1_scores(f1, properties, file_name):
    ax = sns.violinplot(x="Antibiotic", y="F1 Score", data=f1, inner='point', hue='Algorithm', cut=0, scale="count",
                   palette="muted")
    sns.despine()  # Remove the top and right graph lines

    plt.savefig(f'{properties.analysis_dir}{file_name}.jpeg', bbox_inches='tight')
    _free_plot_memory()


def plot_f1_micro_scores(f1, properties, file_name):
    sns.barplot(x="Antibiotic", y="F1 Score", data=f1, hue='Algorithm',
                   palette="muted")
    sns.despine()  # Remove the top and right graph lines
    plt.savefig(f'{properties.analysis_dir}{file_name}.jpeg', bbox_inches='tight')
    _free_plot_memory()
