# -*- coding: utf-8 -*-
"""
Created on Mar 24 6:48 PM 2021

@author: Cory Kromer-Edwards

Main file. Code execution starts here.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import preprocessing
import forests
import neural_network as nn
import properties
import analysis

DO_GRID_SEARCH = False


def set_columns(df, gene_name):
    df = df.set_axis([i for i in range(len(df.columns))], axis=1)
    df = df.add_prefix(f'{gene_name}_')
    return df


def setup_dirs(prop):
    # Create the preprocessed data directory if it does not exist
    if not os.path.exists(f'{prop.data_dir}{prop.processed_data}'):
        os.mkdir(f'{prop.data_dir}{prop.processed_data}')

    if not os.path.exists(f'{prop.output_dir}'):
        os.mkdir(f'{prop.output_dir}')

    if not os.path.exists(f'{prop.analysis_dir}'):
        os.mkdir(f'{prop.analysis_dir}')

    if not os.path.exists(f'{prop.output_dir}/cv_results/'):
        os.mkdir(f'{prop.output_dir}/cv_results/')

    if not os.path.exists(f'{prop.output_dir}/grid_search/'):
        os.mkdir(f'{prop.output_dir}/grid_search/')

    if not os.path.exists(f'{prop.output_dir}/models/'):
        os.mkdir(f'{prop.output_dir}/models/')

    # Set up folder structure for algorithms
    forests.check_and_create_xgb_folders(prop)
    forests.check_and_create_rf_folders(prop)
    nn.check_and_create_dense_folders(prop)


def do_preprocess(prop):
    # Preprocess data (Already done)
    preprocessing.create_form_three(prop)
    _ = preprocessing.create_label_data(prop)


def get_data(prop):
    # Load and set up data
    ompk35 = pd.read_csv(f"{prop.data_dir}{prop.processed_data}form_3_ompk35.csv", index_col=0)
    ompk36 = pd.read_csv(f"{prop.data_dir}{prop.processed_data}form_3_ompk36.csv", index_col=0)
    ompk37 = pd.read_csv(f"{prop.data_dir}{prop.processed_data}form_3_ompk37.csv", index_col=0)
    labels = pd.read_csv(f"{prop.data_dir}{prop.processed_data}labels.csv", index_col=0)
    mic_set = pd.read_csv(f"{prop.data_dir}{prop.processed_data}mic_set.csv")

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
    return mic_set, x_train, x_test, y_train, y_test, labels


def down_sample(train_x, train_y):
    """
    The MIC classes may be extremely skewed, so the algorithms will struggle to train.
    We will shrink all classes down to the sample sample size (down sample) since we cannot
    upsample as creating the new data (Amino Acid sequences for 3 genes) is not easy.
    """
    # First get the MIC with the minimum amount of data
    mics = train_y.unique()
    min_mic_count = 100000
    for m in mics:
        # print(f"{m}: {(labels_scaled.values == m).sum()}")
        min_mic_count = min(min_mic_count, (train_y.values == m).sum())

    # Since we are doing 5-fold CV, we need some classes to have more than 5 elements
    if min_mic_count <= 5:
        min_mic_count = 15

    chosen_indicies = np.array([])
    for m in mics:
        chosen_indicies = np.append(chosen_indicies,
                                    np.random.choice(train_y[train_y == m].index, min_mic_count,
                                                     replace=False))

    train_y = train_y[chosen_indicies]
    train_x = train_x[train_x.index.isin(chosen_indicies)]
    return train_x, train_y


def do_algorithms(prop, mic_set, x_train, x_test, y_train, y_test, labels):
    roc_auc_all_seaborn = pd.DataFrame([],
                                       columns=['Antibiotic', 'False Positive Rate', 'True Positive Rate', 'Algorithm'])
    f1 = pd.DataFrame([], columns=['Antibiotic', 'F1 Score', 'Algorithm'])
    all_f1_micro = pd.DataFrame([], columns=['Antibiotic', 'F1 Score', 'mic', 'Algorithm'])
    overall_gs_results = pd.DataFrame([], columns=['Antibiotic', 'F1-micro Score', 'min_samples_leaf', 'min_samples_split', 'max_depth'])
    for col in labels.columns:
        print(f'Working on {col}:')

        if DO_GRID_SEARCH:

            # Grid search is more manual, so if you want to do Grid Search later you will need to
            # change the variables, and algorithms, here, in the algorithm's grid_search() function, and in
            # plot_graphs' plot_gs function(s).
            classes = y_train[col].unique()
            classes.sort()
            xgb = forests.GradientForest(prop, col, classes)    # XGBoost
            rf = forests.RandomForest(prop, col)                # Random Forest

            # gs_results = xgb.grid_search(x_train, y_train[col])
            gs_results = rf.grid_search(x_train, y_train[col])
            for _, row in gs_results.iterrows():
                overall_gs_results = overall_gs_results.append({'Antibiotic': col,
                                                                'F1-micro Score': row['mean_test_score'],
                                                                'min_samples_leaf': row['param_min_samples_leaf'],
                                                                'min_samples_split': row['param_min_samples_split'],
                                                                'max_depth': row['param_max_depth']},
                                                               ignore_index=True)
        else:
            classes = labels[col].unique()
            classes.sort()

            # x_train_ds, y_train_ds = down_sample(x_train, y_train[col])
            x_train_ds, y_train_ds = x_train, y_train[col]

            xgb = forests.GradientForest(prop, col, classes)                    # XGBoost
            rf = forests.RandomForest(prop, col)                                # Random Forest
            dense_nn = nn.DenseNN(prop, col, len(x_train.columns), classes)     # Dense Neural Network

            print('\tTraining XGBoost model')
            xgb.train(x_train_ds, y_train_ds)
            print('\tTesting XGBoost model')
            fpr, tpr, roc_auc, f1_scores, f1_micro = xgb.test(x_test, y_test[col])
            print('\tSaving feature importance values')
            xgb.save_feature_importance()
            feature_imp = xgb.get_feature_importance()

            print('\tPlotting analysis graphs')
            analysis.plot_single_average_roc(fpr, tpr, roc_auc, prop, f'xgboost/{col}_micro_average_roc')
            analysis.plot_all_roc(fpr, tpr, roc_auc, prop, f'xgboost/{col}_all_roc', mic_set['MICs'].values)
            analysis.plot_feat_imp_bar_graph(feature_imp, prop, f'xgboost/{col}_feat_importance')

            all_f1_micro = all_f1_micro.append({'Antibiotic': col, 'F1 Score': f1_micro, 'Algorithm': 'Xgboost'},
                                               ignore_index=True)
            for f, t in zip(fpr['micro'], tpr['micro']):
                roc_auc_all_seaborn = roc_auc_all_seaborn.append(
                    {'Antibiotic': col, 'False Positive Rate': f, 'True Positive Rate': t, 'Algorithm': 'Xgboost'},
                    ignore_index=True)

            for mic, score in enumerate(f1_scores):
                f1 = f1.append({'Antibiotic': col, 'F1 Score': score, 'mic': mic, 'Algorithm': 'Xgboost'},
                               ignore_index=True)

            print('\t=========================================================================')

            # Random Forest
            print('\tTraining Random Forest model')
            rf.train(x_train_ds, y_train_ds)
            print('\tTesting Random Forest model')
            fpr, tpr, roc_auc, f1_scores, f1_micro = rf.test(x_test, y_test[col])
            print('\tSaving feature importance values')
            rf.save_feature_importance()
            feature_imp = rf.get_feature_importance()

            print('\tPlotting analysis graphs')
            analysis.plot_single_average_roc(fpr, tpr, roc_auc, prop, f'rf/{col}_micro_average_roc')
            analysis.plot_all_roc(fpr, tpr, roc_auc, prop, f'rf/{col}_all_roc', mic_set['MICs'].values)
            analysis.plot_feat_imp_bar_graph(feature_imp, prop, f'rf/{col}_feat_importance')

            all_f1_micro = all_f1_micro.append({'Antibiotic': col, 'F1 Score': f1_micro, 'Algorithm': 'Random Forest'},
                                               ignore_index=True)

            for f, t in zip(fpr['micro'], tpr['micro']):
                roc_auc_all_seaborn = roc_auc_all_seaborn.append(
                    {'Antibiotic': col, 'False Positive Rate': f, 'True Positive Rate': t, 'Algorithm': 'Random Forest'},
                    ignore_index=True)

            for mic, score in enumerate(f1_scores):
                f1 = f1.append({'Antibiotic': col, 'F1 Score': score, 'mic': mic, 'Algorithm': 'Random Forest'},
                               ignore_index=True)

            print('\t=========================================================================')

            print('\tTraining Dense NN model')
            dense_nn.train(x_train_ds, y_train_ds)
            print('\tTesting Dense NN model')
            fpr, tpr, roc_auc, f1_scores, f1_micro = dense_nn.test(x_test, y_test[col])

            print('\tPlotting analysis graphs')
            analysis.plot_single_average_roc(fpr, tpr, roc_auc, prop, f'dense/{col}_micro_average_roc')
            analysis.plot_all_roc(fpr, tpr, roc_auc, prop, f'dense/{col}_all_roc', mic_set['MICs'].values)
            analysis.plot_feat_imp_bar_graph(feature_imp, prop, f'dense/{col}_feat_importance')

            all_f1_micro = all_f1_micro.append({'Antibiotic': col, 'F1 Score': f1_micro, 'Algorithm': 'Dense NN'},
                                               ignore_index=True)
            for f, t in zip(fpr['micro'], tpr['micro']):
                roc_auc_all_seaborn = roc_auc_all_seaborn.append(
                    {'Antibiotic': col, 'False Positive Rate': f, 'True Positive Rate': t, 'Algorithm': 'Dense NN'},
                    ignore_index=True)

            for mic, score in enumerate(f1_scores):
                f1 = f1.append({'Antibiotic': col, 'F1 Score': score, 'mic': mic, 'Algorithm': 'Dense NN'},
                               ignore_index=True)

        if DO_GRID_SEARCH:
            print('Saving Grid Search results and plotting')
            overall_gs_results.to_csv(f'{prop.output_dir}/grid_search/rf.csv')  # Change this line when changing algorithms
            analysis.plot_gs_results(overall_gs_results, prop)
        else:
            print("Plotting F1")
            analysis.plot_f1_scores(f1, prop, "f1_scores_by_mic")
            f1.to_csv(f'{prop.output_dir}f1_test_scores.csv', index=False)
            analysis.plot_f1_micro_scores(all_f1_micro, prop, "f1_micro_scores")
            all_f1_micro.to_csv(f'{prop.output_dir}f1_test_micro_scores.csv', index=False)
            print("Plotting All ROC curves")
            analysis.plot_all_average_roc(roc_auc_all_seaborn, prop, "all_micro_average_roc")



def should_do_preprocessing(prop):
    """Returns False if directory structure is set up and preprocessing is done. True otherwise"""
    return not (os.path.exists(f'{prop.data_dir}{prop.processed_data}') or
           os.path.exists(f"{prop.data_dir}{prop.processed_data}form_3_ompk35.csv"))


def main():
    prop = properties.load_properties(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.properties"))

    setup_dirs(prop)

    if should_do_preprocessing(prop):
        print("Setting up directory structure")

        print("Doing preprocessing")
        do_preprocess(prop)

    print("Getting data")
    mic_set, x_train, x_test, y_train, y_test, labels = get_data(prop)
    print("Starting algorithms")
    do_algorithms(prop, mic_set, x_train, x_test, y_train, y_test, labels)


if __name__ == '__main__':
    main()
