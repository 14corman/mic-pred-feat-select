# -*- coding: utf-8 -*-
"""
Created on Mar 24 8:09 PM 2021

@author: Cory Kromer-Edwards

This file will be responsible for processing label data.
"""

import pandas as pd
import numpy as np


def get_mics(col):
    def try_extract(x):
        if isinstance(x, str):
            return float(x.lstrip('<=').lstrip('>'))
        elif np.isnan(x):
            return -1.0
        else:
            return float(x)
    return pd.Series([try_extract(x) for x in col], dtype=float)


def encode_mics(col, set_mics=[]):
    return pd.Series([set_mics.index(x) for x in col], dtype=int)


def create_label_data(properties):
    """Create the processed labels. Return the set of MICs found."""
    ompk35 = pd.read_csv(f'{properties.data_dir}{properties.main_data}antibiotics_OMPK35.tsv', sep='\t', index_col=0)
    ompk36 = pd.read_csv(f'{properties.data_dir}{properties.main_data}antibiotics_OMPK36.tsv', sep='\t', index_col=0)
    ompk37 = pd.read_csv(f'{properties.data_dir}{properties.main_data}antibiotics_OMPK37.tsv', sep='\t', index_col=0)
    labels_df = pd.concat([ompk35, ompk36, ompk37], axis=1)                 # Combine all genes together
    labels_df = labels_df[labels_df.index != 'consensus']                   # Remove consensus row
    labels_df = labels_df[labels_df.index != 'reference']                   # Remove reference row
    labels_df = labels_df.apply(get_mics, axis=1, result_type='broadcast')  # Remove characters and make all values floats

    # Get the set of MIC values being used in study
    set_mics = list(set(np.concatenate(labels_df.values)))
    set_mics.sort()
    labels_df = labels_df.apply(encode_mics, axis=1, result_type='broadcast', set_mics=set_mics)

    # ==================Remove and chose actual antibiotics to include====================================
    labels_df = labels_df.sample(n=5, axis='columns')
    labels_df.columns = ['Antibiotic_1', 'Antibiotic_2', 'Antibiotic_3', 'Antibiotic_4', 'Antibiotic_5']

    labels_df.to_csv(f'{properties.data_dir}{properties.processed_data}labels.csv')
    pd.DataFrame({"MICs": set_mics}).to_csv(f'{properties.data_dir}{properties.processed_data}mic_set.csv', index=False)
    return set_mics


def decode_mic(mic_index, set_mics):
    """Take in an MIC index and return NaN if -1, otherwise return the MIC at that index."""
    return set_mics[mic_index]
