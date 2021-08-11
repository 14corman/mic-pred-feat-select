# -*- coding: utf-8 -*-
"""
Created on Aug 01 6:31 AM 2021

@author: Cory Kromer-Edwards

"""

from nn import load_best_model
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

DATA_PATH = "../data/processed/"
OUTPUT_PATH = "../output/"
ID_TO_ANTIBIOTIC = {
    1: "Ceftolozane-tazobactam",
    3: "Meropenem",
    4: "Piperacillin-tazobactam",
    5: "Cefepime",
    6: "Ceftobiprole",
    7: "Ampicillin-sulbactam",
    8: "Imipenem",
    9: "Aztreonam",
    10: "Ceftriaxone",
    11: "Ceftaroline",
    12: "Doripenem",
    13: "Ceftazidime-avibactam",
    15: "Ceftazidime",
}


def mic_within_plus_minus_1(y_true, y_pred):
    if y_true - 1 == y_pred:
        return 1
    elif y_true + 1 == y_pred:
        return 1
    elif y_true == y_pred:
        return 1
    else:
        return 0


def log_to_conc_mic(mic):
    return 2**mic


def get_s_r_breakpoints(antibiotic_id):
    """Get CLSI 2021 breakpoints for antibiotics"""
    sr_values = {
        1: (2., 8.),
        3: (1, 4.),
        4: (16., 128.),
        5: (2., 16.),
        6: (4., 8.),    # THIS IS A GUESS. Actual breakpoints not found (Ceftobiprole)
        7: (8., 32.),
        8: (1., 4.),
        9: (4., 16.),
        10: (1., 4.),
        11: (0.5, 2.),
        12: (1., 4.),
        13: (8., 16.),
        15: (4., 16.),
    }

    return sr_values.get(antibiotic_id)


def create_error_report():
    print("Loading model")
    model = load_best_model()
    print("Model loaded")
    data_collection = dict()
    total_error = {
        "minor_error_count": 0,
        "major_error_count": 0,
        "very_major_error_count": 0,
        "num_samples": 0
    }
    with open(f"{DATA_PATH}nn_test") as f:
        for line in f:
            mic, xs = line.split(":")
            mic = int(mic)
            x = np.array(xs.split(","), dtype='float32')
            pred = model.predict(np.array([x, ]))[0][0]
            pred = round(pred)
            for antibiotic_id in ID_TO_ANTIBIOTIC.keys():
                if x[antibiotic_id] != 0:
                    if ID_TO_ANTIBIOTIC[antibiotic_id] not in data_collection:
                        data_collection[ID_TO_ANTIBIOTIC[antibiotic_id]] = {
                            "minor_error_count": 0,
                            "major_error_count": 0,
                            "very_major_error_count": 0,
                            "num_samples": 0
                        }

                    antibiotic = data_collection[ID_TO_ANTIBIOTIC[antibiotic_id]]
                    if mic_within_plus_minus_1(mic, pred) == 0:
                        s, r = get_s_r_breakpoints(antibiotic_id)
                        conv_mic = log_to_conc_mic(mic)
                        conv_pred = log_to_conc_mic(pred)
                        if conv_mic >= r and conv_pred <= s:
                            antibiotic["very_major_error_count"] += 1
                            total_error["very_major_error_count"] += 1

                        #  actual=S and pred=I     actual=I and pred=R      actual=R and pred=I     actual=I and pred=S
                        elif (r > conv_pred > s >= conv_mic) or (s < conv_mic < r <= conv_pred) or \
                                (conv_mic >= r > conv_pred > s) or (r > conv_mic > s >= conv_pred):
                            antibiotic["minor_error_count"] += 1
                            total_error["minor_error_count"] += 1
                        elif conv_mic <= s and conv_pred >= r:
                            antibiotic["major_error_count"] += 1
                            total_error["major_error_count"] += 1

                    antibiotic["num_samples"] += 1
                    total_error["num_samples"] += 1

                    break

    print("Finished collecting test data\nBuilding data DataFrame")
    error_df = pd.DataFrame(columns=["Antibiotic", "Error type", "count", "average"])
    for antibiotic, ant_data in data_collection.items():
        for error, count in ant_data.items():
            if error != "num_samples":
                error_df = error_df.append({
                    "Antibiotic": antibiotic,
                    "Error type": error,
                    "count": count,
                    "average": count / ant_data["num_samples"]
                }, ignore_index=True)

    for error in ["very_major_error_count", "minor_error_count", "major_error_count"]:
        error_df = error_df.append({
            "Antibiotic": "Total",
            "Error type": error,
            "count": total_error[error],
            "average": total_error[error] / total_error["num_samples"]
        }, ignore_index=True)

    print("Creating CSV sheet")
    error_df.pivot("Antibiotic", "Error type", ["count", "average"])\
            .to_csv(f'{OUTPUT_PATH}/error_rates.csv')


def create_mic_heatmap():
    print("Loading model")
    model = load_best_model()
    print("Model loaded")
    data_collection = dict()
    with open(f"{DATA_PATH}nn_test") as f:
        for line in f:
            mic, xs = line.split(":")
            mic = int(mic)
            x = np.array(xs.split(","), dtype='float32')
            pred = model.predict(np.array([x, ]))[0][0]
            pred = round(pred)
            for antibiotic_id in ID_TO_ANTIBIOTIC.keys():
                if x[antibiotic_id] != 0:
                    if ID_TO_ANTIBIOTIC[antibiotic_id] not in data_collection:
                        data_collection[ID_TO_ANTIBIOTIC[antibiotic_id]] = dict()

                    antibitoic = data_collection[ID_TO_ANTIBIOTIC[antibiotic_id]]
                    if mic not in antibitoic:
                        antibitoic[mic] = {"y_pred": [], "y_true": []}

                    antibitoic[mic]["y_pred"].append(mic_within_plus_minus_1(mic, pred))
                    antibitoic[mic]["y_true"].append(1)
                    break

    print("Finished collecting test data\nBuilding data DataFrame")
    collection_df = pd.DataFrame(columns=["Antibiotic", "Minimum Inhibitory Concentration", "f1", "count"])
    for antibiotic, ant_data in data_collection.items():
        for mic, mic_data in ant_data.items():
            collection_df = collection_df.append({
                "Antibiotic": antibiotic,
                "Minimum Inhibitory Concentration": log_to_conc_mic(mic),
                "f1": f1_score(mic_data["y_true"], mic_data["y_pred"], average='binary', zero_division='warn'),
                "count": len(mic_data["y_true"])
            }, ignore_index=True)

    print("Creating heatmap and saving")
    mic_f1_df = collection_df.pivot("Antibiotic", "Minimum Inhibitory Concentration", "f1")
    mic_count_df = collection_df.pivot("Antibiotic", "Minimum Inhibitory Concentration", "count")
    sns.heatmap(mic_f1_df, annot=mic_count_df, fmt="d", cbar_kws={'label': 'Within 1 2-fold dilution F1 score'}, annot_kws={"fontsize": 8})
    sns.despine()  # Remove the top and right graph lines
    plt.savefig(f'{OUTPUT_PATH}/mic_prediction_heatmap.pdf', bbox_inches='tight')


if __name__ == "__main__":
    create_mic_heatmap()
    create_error_report()
