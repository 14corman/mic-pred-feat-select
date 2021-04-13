# -*- coding: utf-8 -*-
"""
Created on Mar 24 6:48 PM 2021

@author: Cory Kromer-Edwards

Main file. Code execution starts here.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

import preprocessing
import forests
import properties
import analysis


def set_columns(df, gene_name):
    df = df.set_axis([i for i in range(len(df.columns))], axis=1)
    df = df.add_prefix(f'{gene_name}_')
    return df


if __name__ == '__main__':
    properties = properties.load_properties(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.properties"))

    # Create the preprocessed data directory if it does not exist
    if not os.path.exists(f'{properties.data_dir}{properties.processed_data}'):
        os.mkdir(f'{properties.data_dir}{properties.processed_data}')

    if not os.path.exists(f'{properties.output_dir}'):
        os.mkdir(f'{properties.output_dir}')

    if not os.path.exists(f'{properties.analysis_dir}'):
        os.mkdir(f'{properties.analysis_dir}')

    if not os.path.exists(f'{properties.output_dir}/cv_results/'):
        os.mkdir(f'{properties.output_dir}/cv_results/')

    if not os.path.exists(f'{properties.output_dir}/models/'):
        os.mkdir(f'{properties.output_dir}/models/')

    forests.check_and_create_xgb_folders(properties)

    preprocessing.create_form_three(properties)
    set_mics = preprocessing.create_label_data(properties)

    # Load and set up data
    ompk35 = pd.read_csv("../data/processed/form_3_ompk35.csv", index_col=0)
    ompk36 = pd.read_csv("../data/processed/form_3_ompk36.csv", index_col=0)
    ompk37 = pd.read_csv("../data/processed/form_3_ompk37.csv", index_col=0)
    labels = pd.read_csv("../data/processed/labels.csv", index_col=0)
    mic_set = pd.read_csv("../data/processed/mic_set.csv")

    # Set column names to distinguish between genes
    ompk35 = set_columns(ompk35, 'ompk35')
    ompk36 = set_columns(ompk36, 'ompk36')
    ompk37 = set_columns(ompk37, 'ompk37')

    # Concatenate genes together
    form_3 = pd.concat([ompk35, ompk36, ompk37], axis=1, join='inner')

    # Make sure labels and form_3 match
    labels = labels[labels.index.isin(form_3.index)]
    labels = labels.sort_index()
    form_3 = form_3.sort_index()

    x_train, x_test, y_train, y_test = train_test_split(form_3, labels, test_size=0.2, random_state=642)

    fpr_all = dict()
    tpr_all = dict()
    roc_auc_all = dict()
    f1 = pd.DataFrame([], columns=['antibiotic', 'f1 score', 'algorithm'])
    all_f1_micro = pd.DataFrame([], columns=['antibiotic', 'f1 score', 'mic', 'algorithm'])
    for col in labels.columns:
        print(f'Working on {col}:')
        print('\tTraining XGBoost model')
        xgb = forests.GradientForest(properties, col, len(mic_set["MICs"]))
        xgb.train(x_train, y_train[col])
        print('\tTesting XGBoost model')
        fpr, tpr, roc_auc, f1_scores, f1_micro = xgb.test(x_test, y_test[col])
        print('\tSaving feature importance values')
        xgb.save_feature_importance()
        feature_imp = xgb.get_feature_importance()

        print('\tPlotting analysis graphs')
        analysis.plot_single_average_roc(fpr, tpr, roc_auc, properties, f'{col}_micro_average_roc')
        analysis.plot_all_roc(fpr, tpr, roc_auc, properties, f'{col}_all_roc', mic_set['MICs'].values)
        analysis.plot_feat_imp_bar_graph(feature_imp, properties, f'{col}_feat_importance')

        fpr_all[col] = fpr['micro']
        tpr_all[col] = tpr['micro']
        roc_auc_all[col] = roc_auc['micro']
        all_f1_micro = all_f1_micro.append({'antibiotic': col, 'f1 score': f1_micro, 'algorithm': 'Xgboost'}, ignore_index=True)
        for mic, score in enumerate(f1_scores):
            f1 = f1.append({'antibiotic': col, 'f1 score': score, 'mic': mic, 'algorithm': 'Xgboost'}, ignore_index=True)

    analysis.plot_all_average_roc(fpr_all, tpr_all, roc_auc_all, properties, 'all_micro_average_roc')
    analysis.plot_f1_scores(f1, properties, "f1_scores_by_mic")
    f1.to_csv(f'{properties.output_dir}f1_test_scores.csv', index=False)
    analysis.plot_f1_micro_scores(all_f1_micro, properties, "f1_micro_scores")
    all_f1_micro.to_csv(f'{properties.output_dir}f1_test_micro_scores.csv', index=False)
