# -*- coding: utf-8 -*-
"""
Created on Jun 02 6:47 AM 2021

@author: Cory Kromer-Edwards

Standard Machine Learning functions that need to be used by all algorithms.

"""

from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd


# noinspection PyAttributeOutsideInit
def f1_eval(y_pred, dtrain):
    """Custom f1 score function for XGBoost"""
    y_true = dtrain.get_label()
    err = f1_score(y_true, np.argmax(y_pred, axis=1), average='weighted')
    return 'f1', err


def scale_labels(x, classes=[]):
    """Scaling down labels to be [0, num_classes)"""
    return classes.index(x)  # np.where(classes == x)[0][0]


def one_hot(x, num_classes=0):
    """Turn an element in a column into a one-hot encoded list"""
    # l = [0 for _ in range(num_classes)]
    # l[x] = 1
    # return np.asarray(l)
    return tf.keras.utils.to_categorical(x, num_classes=num_classes)


def build_output_nn(dataset, features, num_classes):
    """
    Build and compile the output Dense NN for an algorithm. Each algorithm will be using the same Dense NN
        architecture, so they will all be calling this function.
    :param dataset: Dataset (converted from dataframe using "df_to_dataset") that will be used to created normalized layers
    :param features: List of feature names (columns) in dataset's original dataframe
    :param num_classes: Number of classes/MICs that are being predicted for
    :return: A compiled Dense NN model
    """
    all_inputs = []
    encoded_features = []

    # Numeric features.
    for header in features:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, dataset)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(1000, activation="relu")(all_features)
    x = tf.keras.layers.Dense(1000, activation="relu")(x)
    x = tf.keras.layers.Dense(1000, activation="relu")(x)
    x = tf.keras.layers.Dense(500, activation="relu")(x)
    x = tf.keras.layers.Dense(250, activation="relu")(x)
    x = tf.keras.layers.Dense(100, activation="relu")(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["categorical_accuracy"])

    # model.compile(optimizer='sgd', loss='categorical_crossentropy',
    #               metrics=['categorical_crossentropy', 'categorical_accuracy'])
    return model


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    """
    A utility method to create a tf.data dataset from a Pandas Dataframe.
    Code taken from: https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
    :param dataframe: Pandas dataset (Must have labels stored in column named "target")
    :param shuffle: (optional) Whether the layer should be shuffed (default: true)
    :param batch_size: (optional) Batch size for the layer (default: 32)
    :return: Tensorflow dataset batched
    """
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    columns = [f"mic_{i}" for i in range(len(labels.iloc[0]))]
    labels = pd.DataFrame(labels.to_list(), columns=columns)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def get_normalization_layer(name, dataset):
    """
    Create a normalized layer for a specific feature. The layer will be used as input.
    Code taken from: https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
    :param name: Name of feature to make a layer for
    :param dataset: Dataset (converted from dataframe) that has the feature in it
    :return: A tensorflow normalized layer class
    """
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization()

    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    """
    Create a categorical layer for a specific feature. The layer will be used as input.
    Code taken from: https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
    :param name: Name of feature to make a layer for
    :param dataset: Dataset (converted from dataframe) that has the feature in it
    :param dtype: The datatype of the feature (Either use 'string' or anything else for other dtypes)
    :param max_tokens: The maximum size of the vocabulary for this layer. If None, there
                        is no cap on the size of the vocabulary. Note that this size
                        includes the OOV and mask tokens. Default to None.
    :return: A tensorflow categorical layer class
    """
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    if dtype == 'string':
        encoder = preprocessing.CategoryEncoding(max_tokens=index.vocabulary_size())
    else:
        encoder = preprocessing.CategoryEncoding(max_tokens=len(index.get_vocabulary()))

    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))
