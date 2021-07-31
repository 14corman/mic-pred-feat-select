# -*- coding: utf-8 -*-
"""
Created on Jul 28 8:40 PM 2021

@author: Cory Kromer-Edwards

Original code for some functions below came from:
https://www.machinecurve.com/index.php/2020/04/06/using-simple-generators-to-flow-data-from-file-with-keras/

## Create NN input files
This file will go through taking the train.libsvm file and test.libsvm file and create nn_train, nn_val, and
nn_test files out of it. Those files will be formatted to make it easy to create batches to train and test
the neural network.

nn_train, nn_val, and nn_test file formats:
{mic}:{feat_1_count},{feat_2_count},{feat_2_count},...

The features are ordered by the FEATURES_TO_USE list.

Update NUM_TRAINING_ROWS, NUM_VALIDATION_ROWS, NUM_TEST_ROWS accordingly using the console output after
the files have been made.

## Train NN
The next step will be to train the NN. It will outputs checkpoint models in the format:
best_nn_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5

After training, delete all by the last hdf5 file, take note of its name, and set it as BEST_NN_MODEL.

## Test NN
After you set BEST_NN_MODEL, you can run the test method to get an F1-micro score that allows
predictions to be within +-1 2-fold dilution of the actual MIC.

## Order
Generating the files and Training function can be run back-to-back.
Then, you need to set BEST_NN_MODEL and only run test function.

"""

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import tensorflow
import random
import re
import sys

# Num rows
NUM_TRAINING_ROWS = 40385
NUM_VALIDATION_ROWS = 10085
NUM_TEST_ROWS = 12630   # Not used, but good to know
BATCH_SIZE = 250
DATA_PATH = "../data/processed/"
OUTPUT_PATH = "../output/"

# Features below are in order of importance with higher importance features at the top
# If you retrained XGBoost, then you will have to recreate this list.
FEATURES_TO_USE = [
    0,
    524804,
    1,
    524812,
    524802,
    524805,
    524808,
    524803,
    524810,
    524800,
    524809,
    524806,
    524811,
    524801,
    96031,
    524807,
    3,
    2,
    6,
    5,
    464895,
    4,
    9,
    7,
    16,
    252593,
    185209,
    15,
    436636
]

# Collected after training is done. Will be loaded and tested on.
BEST_NN_MODEL = f"{OUTPUT_PATH}best_nn_model.epoch440-loss0.98.hdf5"


# Class code originally came from: https://stackoverflow.com/a/2555047/9659107
# This code was run in Python 3.7, so no :=
class REMatcher(object):
    def __init__(self, regex):
        self.regex = regex

    def match(self, match_string):
        self.rematch = re.match(self.regex, match_string)
        return bool(self.rematch)

    def group(self, i):
        return self.rematch.group(i)


def generate_nn_files():
    num_train_lines = 0
    num_val_lines = 0
    with open(f"{DATA_PATH}train.libsvm") as input_file, \
            open(f"{DATA_PATH}nn_train", "w") as train_file, \
            open(f"{DATA_PATH}nn_validation", "w") as val_file:
        for line in input_file:
            line_split = line.split(" ")
            mic = int(line_split[0]) - 13
            x = []
            for feat in FEATURES_TO_USE:
                m = REMatcher(f".+{feat}:(\\d+).*")  # Check if feature is in libsvm formatted string
                if m.match(line):
                    if feat > 524799:  # Features after this id are antibiotics
                        x.append("1")
                    else:
                        x.append(str(m.group(1)))
                else:
                    if feat > 524799:  # Features after this id are antibiotics
                        x.append("0")
                    else:
                        x.append("0")

            if random.random() < 0.8:
                train_file.write(f"{mic}:{','.join(x)}\n")
                num_train_lines += 1
            else:
                val_file.write(f"{mic}:{','.join(x)}\n")
                num_val_lines += 1

    num_test_lines = 0
    with open(f"{DATA_PATH}test.libsvm") as input_file, \
            open(f"{DATA_PATH}nn_test", "w") as test_file:
        for line in input_file:
            line_split = line.split(" ")
            mic = int(line_split[0]) - 13
            x = []
            for feat in FEATURES_TO_USE:
                m = REMatcher(f".+{feat}:(\\d+).*")  # Check if feature is in libsvm formatted string
                if m.match(line):
                    if feat > 524799:  # Features after this id are antibiotics
                        x.append("1")
                    else:
                        x.append(str(m.group(1)))
                else:
                    if feat > 524799:  # Features after this id are antibiotics
                        x.append("0")
                    else:
                        x.append("0")

            test_file.write(f"{mic}:{','.join(x)}\n")
            num_test_lines += 1

    print(f"Number of rows in training file: {num_train_lines}")
    print(f"Number of rows in validation file: {num_val_lines}")
    print(f"Number of rows in test file: {num_test_lines}")
    print("If you have restarted project with new data, then you will need to update top of nn.py file with above values.")


# Load data
def generate_arrays_from_file(path, batch_size):
    inputs = []
    targets = []
    batch_count = 0
    while True:
        with open(path) as f:
            for line in f:
                mic, xs = line.split(":")
                inputs.append(np.array(xs.split(","), dtype='float32'))
                targets.append(mic)
                batch_count += 1
                if batch_count > batch_size:
                    X = np.array(inputs, dtype='float32')
                    y = np.array(targets, dtype='float32')
                    yield (X, y)
                    inputs = []
                    targets = []
                    batch_count = 0


def train_model():
    # Create the model
    model = Sequential()
    model.add(Dense(50, input_dim=len(FEATURES_TO_USE), activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    # Compile the model
    model.compile(loss='mean_absolute_error',
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=['mean_squared_error'])

    best_model_path = OUTPUT_PATH + "best_nn_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath=best_model_path,
                                 monitor="val_loss", save_best_only=True, mode="min")

    callbacks = [checkpoint]
    
    # Fit data to model
    history = model.fit(generate_arrays_from_file(f"{DATA_PATH}nn_train", BATCH_SIZE),
                        steps_per_epoch=NUM_TRAINING_ROWS / BATCH_SIZE, epochs=500,
                        validation_data=generate_arrays_from_file(f"{DATA_PATH}nn_validation", BATCH_SIZE),
                        validation_steps=NUM_VALIDATION_ROWS / BATCH_SIZE,
                        callbacks=callbacks)

    tensorflow.keras.utils.plot_model(model, show_shapes=True, rankdir="LR",
                              to_file=f'{OUTPUT_PATH}models/nn_dense.png')

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'{OUTPUT_PATH}nn_model_training_history.pdf')

    # Epoch 369/500
    # 161/161 [==============================] - 1s 5ms/step - loss: 0.9202 - mean_squared_error: 3.1195 - val_loss: 0.9817 - val_mean_squared_error: 3.4280


def set_plus_minus_1(y_true, y_pred):
    if y_true - 1 == y_pred:
        return y_true
    elif y_true + 1 == y_pred:
        return y_true
    else:
        return y_pred


def test_nn_model():
    model = load_model(BEST_NN_MODEL)

    y_true = []
    y_pred = []
    with open(f"{DATA_PATH}nn_test") as f:
        for line in f:
            mic, xs = line.split(":")
            mic = int(mic)
            x = np.array(xs.split(","), dtype='float32')
            pred = model.predict(np.array([x,]))[0][0]
            pred = round(pred)
            pred = set_plus_minus_1(mic, pred)
            y_pred.append(pred)
            y_true.append(mic)

    f1 = f1_score(y_true, y_pred, average='micro', zero_division='warn')

    print(f"NN model F1 score: {f1}")
    # -------------Rounding is better-------
    # NN model F1 score: 0.8025336500395883   (ROUNDING with +-1 2-fold dilution)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise ValueError("No input given. Expected 1 of ['files', 'train', 'test']")

    run_command = sys.argv[1]
    if run_command == "files":
        generate_nn_files()
    elif run_command == "train":
        train_model()
    elif run_command == "test":
        test_nn_model()
    else:
        raise ValueError(f"Unknown input '{run_command}'. Expected 1 of ['files', 'train', 'test']")
