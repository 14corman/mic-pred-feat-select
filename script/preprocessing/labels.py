# -*- coding: utf-8 -*-
"""
Created on Mar 24 8:09 PM 2021

@author: Cory Kromer-Edwards

This file will be responsible for processing label data.
"""

import pandas as pd
import numpy as np

possible_mics = [0.001, 0.003, 0.007, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1., 2., 4., 8., 16., 32., 64., 128.,
                 256., 512., 1024.]


def encode_mics(col):
    def try_extract(x):
        if isinstance(x, str):
            return possible_mics.index(float(x.lstrip('<=').lstrip('>')))
        elif np.isnan(x):
            return -1
        else:
            return possible_mics.index(float(x))
    return pd.Series([try_extract(x) for x in col], dtype=int)


def create_label_data(tsv_name, properties):
    """Create the processed labels."""
    labels_df = pd.read_csv(f'{properties.data_dir}{tsv_name}', sep='\t', index_col=0)
    labels_df = labels_df.apply(encode_mics, axis=1, result_type='broadcast')
    labels_df.to_csv(f'{properties.data_dir}processed/labels.csv')


def decode_mic(mic_index):
    """Take in an MIC index and return NaN if -1, otherwise return the MIC at that index."""
    if mic_index == -1:
        return np.nan
    else:
        return possible_mics[mic_index]
