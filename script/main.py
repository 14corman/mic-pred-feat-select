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
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import OneSidedSelection
from imblearn.combine import SMOTEENN

import preprocessing
import forests
import neural_network as nn
import properties
import analysis

DO_GRID_SEARCH = False
DO_RESAMPLE = False

all_roc_auc = pd.DataFrame([], columns=['Antibiotic', 'False Positive Rate', 'True Positive Rate', 'Algorithm'])
f1 = pd.DataFrame([], columns=['Antibiotic', 'F1 Score', 'Algorithm'])
all_f1_micro = pd.DataFrame([], columns=['Antibiotic', 'F1 Score', 'mic', 'Algorithm'])


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

    if not os.path.exists(f'{prop.analysis_dir}/n_training_error/'):
        os.mkdir(f'{prop.analysis_dir}/n_training_error/')

    # Set up folder structure for algorithms
    forests.check_and_create_xgb_folders(prop)
    forests.check_and_create_rf_folders(prop)
    nn.check_and_create_dense_folders(prop)
    nn.check_and_create_lstm_folders(prop)


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

    # Get Train/Test split
    test_labels = labels.sample(frac=0.2, random_state=643)
    train_labels = labels.drop(test_labels.index)

    # Each gene may not have every isolate, so ignore error if isolate does not exist
    ompk35_train = ompk35.drop(test_labels.index, errors='ignore')
    ompk35_test = ompk35.drop(train_labels.index, errors='ignore')

    ompk36_train = ompk36.drop(test_labels.index, errors='ignore')
    ompk36_test = ompk36.drop(train_labels.index, errors='ignore')

    ompk37_train = ompk37.drop(test_labels.index, errors='ignore')
    ompk37_test = ompk37.drop(train_labels.index, errors='ignore')

    # Combine train and test gene data
    ompk_train = [ompk35_train, ompk36_train, ompk37_train]
    ompk_test = [ompk35_test, ompk36_test, ompk37_test]

    return mic_set, ompk_train, ompk_test, train_labels, test_labels, labels


def down_sample(train_x, train_y):
    """
    The MIC classes may be extremely skewed, so the algorithms will struggle to train.
    We will shrink all classes down to the sample sample size (down sample) since we cannot
    upsample as creating the new data (Amino Acid sequences for 3 genes) is not easy.
    """
    new_x = []
    new_y = []
    # undersample = TomekLinks()
    undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=25, random_state=59103)
    # undersample = SMOTEENN(n_jobs=5)
    for ompk_train in train_x:
        gene_labels = train_y.filter(items=list(ompk_train.index), axis=0)
        if DO_RESAMPLE:
            from collections import Counter
            print(Counter(gene_labels))
            tmp_x, tmp_y = undersample.fit_resample(ompk_train, gene_labels)
            kept_indicies = gene_labels.iloc[undersample.sample_indices_].index
            tmp_x.set_index(kept_indicies, inplace=True)    # pd.DataFrame
            tmp_y.index = kept_indicies                     # pd.Series
            print(Counter(tmp_y))
            print()
        else:
            tmp_x, tmp_y = ompk_train, gene_labels

        new_x.append(tmp_x)
        new_y.append(tmp_y)

    return new_x, new_y


# DEPRECATED
def remove_all_same_cols(train_x):
    """
    Remove all Amino Acid positions (columns) that are all the same Amino Acid.
    IE: the column only has 1 unique value in it
    Code taken from: https://stackoverflow.com/a/39658662/9659107

    This only removed a few columns and the accuracy did not go up.
    :param train_x:
    :return:
    """
    new_x = []
    for ompk_train in train_x:
        nunique = ompk_train.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        new_x.append(ompk_train.drop(cols_to_drop, axis=1))

    return new_x


def collect_results(fpr, tpr, roc_auc, mic_set, f1_micro, f1_scores, prop, antibiotic, algorithm, folder, feature_imp=None):
    global all_f1_micro, all_roc_auc, f1
    print('\tPlotting analysis graphs')
    for i, output in enumerate(["ompk35", "ompk36", "ompk37", "nn_output"]):
        analysis.plot_single_average_roc(fpr[i], tpr[i], roc_auc[i], prop, f'{folder}/{antibiotic}_{output}_micro_average_roc')
        analysis.plot_all_roc(fpr[i], tpr[i], roc_auc[i], prop, f'{folder}/{antibiotic}_{output}_all_roc',
                              mic_set['MICs'].values)

        if i < 3 and feature_imp is not None:
            analysis.plot_feat_imp_bar_graph(feature_imp[i], prop, f'{folder}/{antibiotic}_{output}_feat_importance')

        if i == 3:
            all_f1_micro = all_f1_micro.append({'Antibiotic': antibiotic, 'F1 Score': f1_micro[i], 'Algorithm': algorithm},
                                               ignore_index=True)

            for f, t in zip(fpr[i]['micro'], tpr[i]['micro']):
                all_roc_auc = all_roc_auc.append(
                    {'Antibiotic': antibiotic, 'False Positive Rate': f, 'True Positive Rate': t, 'Algorithm': algorithm},
                    ignore_index=True)

            for mic, score in enumerate(f1_scores[i]):
                f1 = f1.append({'Antibiotic': antibiotic, 'F1 Score': score, 'mic': mic, 'Algorithm': algorithm},
                               ignore_index=True)


def do_algorithms(prop, mic_set, x_train, x_test, y_train, y_test, labels):
    global all_roc_auc, f1, all_f1_micro

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

            # gs_results = xgb.grid_search(x_train, y_train[col])
            gs_results = xgb.grid_search(x_train, y_train[col])
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
            print(classes)

            x_train_ds, y_train_ds = down_sample(x_train, y_train[col])
            # x_train_ds = remove_all_same_cols(x_train_ds) #DEPRECATED
            # x_train_ds, y_train_ds = x_train, y_train[col]

            xgb = forests.GradientForest(prop, col, classes)        # XGBoost
            num_genes = [len(x_train[i].columns) for i in range(3)]
            # dense_nn = nn.DenseNN(prop, col, num_genes, classes)    # Dense Neural Network
            lstm_nn = nn.LstmNN(prop, col, num_genes, classes)  # LSTM Neural Network

            """
            print('\tTraining XGBoost model')
            xgb.train(x_train_ds, y_train_ds, y_train[col])
            print('\tTesting XGBoost model')
            fpr, tpr, roc_auc, f1_scores, f1_micro = xgb.test(x_test, y_test[col])
            print('\tSaving feature importance values')
            xgb.save_feature_importance()
            feature_imp = xgb.get_feature_importance()

            collect_results(fpr, tpr, roc_auc, mic_set, f1_micro, f1_scores, prop, col, "Xgboost", "xgboost",
                            feature_imp=feature_imp)

            print('\t=========================================================================')

            print('\tTraining Dense NN model')
            error_history = dense_nn.train(x_train_ds, y_train_ds, y_train[col])
            analysis.plot_nn_train_error_history(error_history, prop, f"n_training_error/dense_{col}")
            error_history.to_csv(f'{prop.analysis_dir}n_training_error/dense_{col}.csv', index=False)
            print('\tTesting Dense NN model')
            fpr, tpr, roc_auc, f1_scores, f1_micro = dense_nn.test(x_test, y_test[col])

            collect_results(fpr, tpr, roc_auc, mic_set, f1_micro, f1_scores, prop, col, "Dense NN", "dense")
            print('\t=========================================================================')
            """

            print('\tTraining LSTM NN model')
            error_history = lstm_nn.train(x_train_ds, y_train_ds, y_train[col])
            analysis.plot_nn_train_error_history(error_history, prop, f"n_training_error/lstm_{col}")
            error_history.to_csv(f'{prop.analysis_dir}n_training_error/lstm_{col}.csv', index=False)
            print('\tTesting LSTM NN model')
            fpr, tpr, roc_auc, f1_scores, f1_micro = lstm_nn.test(x_test, y_test[col])

            collect_results(fpr, tpr, roc_auc, mic_set, f1_micro, f1_scores, prop, col, "LSTM NN", "lstm")

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
        analysis.plot_all_average_roc(all_roc_auc, prop, "all_micro_average_roc")


def should_do_preprocessing(prop):
    """Returns False if directory structure is set up and preprocessing is done. True otherwise"""
    return not (os.path.exists(f'{prop.data_dir}{prop.processed_data}') and
           os.path.exists(f"{prop.data_dir}{prop.processed_data}form_3_ompk35.csv"))


def main():
    prop = properties.load_properties(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.properties"))

    print("Setting up directory structure")
    setup_dirs(prop)

    if should_do_preprocessing(prop):
        print("Doing preprocessing")
        do_preprocess(prop)

    print("Getting data")
    mic_set, x_train, x_test, y_train, y_test, labels = get_data(prop)

    print("Generating Pandas report")
    # from pandas_profiling import ProfileReport
    # from pathlib import Path
    # ProfileReport(x_train[0], title="OMPK35 data report", minimal=True).to_file(Path(f"{prop.output_dir}ompk35_report.html"))
    # ProfileReport(x_train[2], title="OMPK36 data report", explorative=True).to_file(Path(f"{prop.output_dir}ompk36_report.html"))
    # ProfileReport(x_train[3], title="OMPK37 data report", explorative=True).to_file(Path(f"{prop.output_dir}ompk37_report.html"))
    # ProfileReport(y_train, title="MIC data report", explorative=True).to_file(Path(f"{prop.output_dir}class_report.html"))

    print("Starting algorithms")
    do_algorithms(prop, mic_set, x_train, x_test, y_train, y_test, labels)


if __name__ == '__main__':
    main()
