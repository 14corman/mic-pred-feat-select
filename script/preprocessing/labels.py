# -*- coding: utf-8 -*-
"""
Created on Mar 24 8:09 PM 2021

@author: Cory Kromer-Edwards

This file will be responsible for processing label data.
"""

import pandas as pd


def drop_symbols(col):
    def try_extract(x):
        if isinstance(x, int) or isinstance(x, float):
            return x
        else:
            return x.lstrip('<=').lstrip('>')
    return pd.Series([try_extract(x) for x in col], dtype=float)


def create_label_data(tsv_name, properties):
    labels_df = pd.read_csv(f'{properties.data_dir}{tsv_name}', sep='\t', index_col=0)
    labels_df = labels_df.apply(drop_symbols, axis=1, result_type='broadcast')
    labels_df.to_csv(f'{properties.data_dir}processed/labels.csv')
