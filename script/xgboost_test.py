# -*- coding: utf-8 -*-
"""
Created on Jul 26 7:07 AM 2021

@author: Cory Kromer-Edwards

This file loads and test our trained XGBoost algorithm and the XGBoost model for KPN saved in:
https://github.com/PATRIC3/mic_prediction/

It will also create an ordered CSV file named feature_importance.csv from our XGBoost model that will go into
the output folder. It will have all features sorted by importance.

"""

import pickle as pkl
from sklearn.metrics import f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_PATH = "../output/"
MODEL_PATH = f"{OUTPUT_PATH}models/"
DATA_PATH = "../data/processed/"
NUM_TEST_FILES = 12


def set_plus_minus_1(y_true, y_pred):
    if y_true - 1 == y_pred:
        return y_true
    elif y_true + 1 == y_pred:
        return y_true
    else:
        return y_pred


def run_test_iteration(iteration, file_path, model):
    dtest = xgb.DMatrix(file_path)
    y_true = dtest.get_label()
    # print(y_true)

    # prediction with test data
    pred = model.predict(dtest)
    pred = pred.round()
    pred = [set_plus_minus_1(i, j) for (i, j) in zip(y_true, pred)]
    # print(pred)

    f1 = f1_score(y_true, pred, average='micro', zero_division='warn')
    print(f"Test file {iteration} is done")

    # dtest could be held in memory long enough for another dtest to be made.
    # This means you may be trying to hold multiple large libsvm files in memory at once which cannot happen.
    # To get around this, before we leave this function we will remove anything asssociated with dtest.
    del y_true
    del dtest
    return f1


def run_our_model_test():
    model_1 = pkl.load(open(f"{MODEL_PATH}xgboost-model", 'rb'))
    print("Model 1 loaded")

    our_f1 = 0.0
    print("Starting testing")
    for i in range(NUM_TEST_FILES):
        our_f1 += run_test_iteration(i, f"{DATA_PATH}test.{i:04}?format=libsvm", model_1)

    our_f1 = our_f1 / NUM_TEST_FILES

    print(f"Our model F1 score: {our_f1}")
    # -------------Rounding is better-------
    # Our model F1 score: 0.7971396402954508  (ROUNDING with +-1 2-fold dilution)


def run_control_model_test():
    control_model = xgb.Booster({})
    control_model.load_model(f"{MODEL_PATH}mic_prediction.model.pkl")
    print("Control model loaded")

    control_f1 = 0.0
    print("Starting testing")
    for i in range(NUM_TEST_FILES):
        with open(f"{DATA_PATH}test.{i:04}") as in_file, open(f"{DATA_PATH}control_test.libsvm", 'w') as out_file:
            for line in in_file:
                line_split = line.split(" ")

                # The extra +13 came from XGBoost not liking labels being <0, so we added 13 to make sure all labels>=0
                mic = int(line_split[0]) - 13
                ant_id = line_split[-1].split(":")[0]
                rest_string = " ".join(line_split[1:-1])

                # Only save rows that use antibiotics that control model was trained on
                if ant_id in ["524800", "524802", "524803", "524805", "524807", "524809", "524810", "524812"]:
                    # Change antibiotic id to match what control  model is looking for
                    if ant_id == "524800":
                        ant_id = "524803"
                    elif ant_id == "524802":
                        ant_id = "524818"
                    elif ant_id == "524803":
                        ant_id = "524802"
                    elif ant_id == "524805":
                        ant_id = "524805"
                    elif ant_id == "524807":
                        ant_id = "524807"
                    elif ant_id == "524809":
                        ant_id = "524808"
                    elif ant_id == "524810":
                        ant_id = "524814"
                    elif ant_id == "524812":
                        ant_id = "524816"

                    out_file.write(f"{mic} {rest_string} {ant_id}:1\n")

        print("Test file created")
        control_f1 += run_test_iteration(i, f"{DATA_PATH}control_test.libsvm", control_model)

    control_f1 = control_f1 / NUM_TEST_FILES

    print(f"Control model F1 score: {control_f1}")
    # -------------Rounding is better-------
    # Control model F1 score: 0.5286073167725044   (ROUNDING with +-1 2-fold dilution)


def build_model_info():
    model = pkl.load(open(f"{MODEL_PATH}xgboost-model", 'rb'))
    xgb.plot_tree(model)

    # The figure scale is too large, so the figure looks blurry when viewed.
    # We need to change the scale
    # Code found: https://github.com/dmlc/xgboost/issues/1725#issuecomment-295059572
    fig = plt.gcf()
    fig.set_size_inches(150, 100)
    fig.savefig(f"{OUTPUT_PATH}model_tree.pdf")
    feat_imprt = pd.DataFrame(list(model.get_fscore().items()),
                 columns=['Kmer', 'Importance']).sort_values('Importance',
                                                             ascending=False)
    feat_imprt.to_csv(f"{OUTPUT_PATH}feature_importance.csv")


if __name__ == "__main__":
    run_our_model_test()
    run_control_model_test()
    build_model_info()
