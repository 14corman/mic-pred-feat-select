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
            if "<=" in x:
                return -2.0
            if ">" in x:
                return -1.0

            return float(x)
            # return float(x.lstrip('<=').lstrip('>'))
        elif np.isnan(x):
            return -3.0
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

    # Merges all columns in each dataframe while keeping the non-null values
    labels_df = ompk35
    labels_df = labels_df.combine_first(ompk36)
    labels_df = labels_df.combine_first(ompk37)
    labels_df = labels_df[labels_df.index != 'consensus']  # Remove consensus row
    labels_df = labels_df[labels_df.index != 'reference']  # Remove reference row

    # Only get Beta-lactam antibiotics
    beta_lactams = ['Penicillin', 'Amoxicillin', 'Ampicillin', 'Piperacillin', 'Oxacillin', 'Mecillinam',
                    'Amoxicillin-clavulanate', 'Ampicillin-sulbactam', 'Aztreonam-avibactam',
                    'Cefepime-tazobactam', 'Cefepime-zidebactam', 'Cefoperazone-sulbactam',
                    'Ceftaroline-avibactam', 'Ceftazidime-avibactam', 'Ceftibuten-clavulanate_fixed_2',
                    'Ceftibuten-clavulanate_2_to_1',
                    'Ceftolozane-tazobactam', 'Meropenem-nacubactam', 'Meropenem-vaborbactam',
                    'Piperacillin-tazobactam', 'Ticarcillin-clavulanate', 'Cefazolin', 'Cefuroxime',
                    'Cefoperazone', 'Ceftazidime', 'Ceftriaxone', 'Cefepime', 'Ceftaroline', 'Ceftobiprole',
                    'Cefoxitin', 'Cefiderocol', 'Cefpodoxime', 'Cefpodoxime_ETX1317', 'Ceftibuten', 'Cefuroxime',
                    'Cephalexin',
                    'Aztreonam', 'Biapenem', 'Doripenem', 'Ertapenem', 'Imipenem', 'Meropenem', 'MeroRPX7009_fixed8',
                    'Razupenem',
                    'Tebipenem', 'Faropenem', 'Sulopenem']

    labels_df = labels_df[[c for c in labels_df.columns if c in beta_lactams]]

    # Filter out columns with >=15% 0's (NaN's/null values)
    # Code found from: https://stackoverflow.com/a/31618099
    max_number_of_nans = len(labels_df.index) * 0.15
    labels_df = labels_df.drop(labels_df.columns[labels_df.apply(lambda col: col.isnull().sum() >= max_number_of_nans)],
                               axis=1)

    labels_df = labels_df.apply(get_mics, axis=1, result_type='broadcast')  # Remove characters and make all values floats

    # Get the set of MIC values being used in study
    # -3.0 = 0 = NaN
    # -2.0 = 1 = MIC had <= (off left side of plate)
    # -1.0 = 2 = MIC had > (off right side of place)
    # Otherwise = MIC
    set_mics = list(set(np.concatenate(labels_df.values)))
    set_mics.sort()
    labels_df = labels_df.apply(encode_mics, axis=1, result_type='broadcast', set_mics=set_mics)

    labels_df.to_csv(f'{properties.data_dir}{properties.processed_data}labels.csv')
    pd.DataFrame({"MICs": set_mics}).to_csv(f'{properties.data_dir}{properties.processed_data}mic_set.csv', index=False)
    return set_mics


def decode_mic(mic_index, set_mics):
    """Take in an MIC index and return NaN if -1, otherwise return the MIC at that index."""
    return set_mics[mic_index]
