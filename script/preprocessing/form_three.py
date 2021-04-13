# -*- coding: utf-8 -*-
"""
Created on Mar 24 8:03 PM 2021

@author: Cory Kromer-Edwards

This preprocessing file looks to create input form 3 where the annotated Amino Acid
sequences will be converted a sequence of numbers corresponding to the character that
occurres in the annotated sequence.
NOTE: It does not rely on the reference sequence
"""

import pandas as pd


def convert(col):
    characters = {
        'A': 0,
        'R': 1,
        'N': 2,
        'D': 3,
        'C': 4,
        'Q': 5,
        'E': 6,
        'G': 7,
        'H': 8,
        'I': 9,
        'L': 10,
        'K': 11,
        'M': 12,
        'F': 13,
        'P': 14,
        'S': 15,
        'T': 16,
        'W': 17,
        'Y': 18,
        'V': 19,
        'B': 20,
        'Z': 21,
        'X': 22,
        '?': 23,
        '-': 24,
        'O': 25
    }

    new_col = []
    for c in col:
        tmp = characters.get(c, None)
        if tmp is None:
            raise ValueError(f'Value {c} does not exist in characters dictionary')

        new_col.append(tmp)

    return new_col


def create_form_three(properties):
    for file_name, gene in zip(['export_msa_75_OMPK35.csv', 'export_msa_75_OMPK36.csv', 'export_msa_75_OMPK37.csv'],
                               ['ompk35', 'ompk36', 'ompk37']):
        # Setting first column as index here allows for column wise apply later
        msa_df = pd.read_csv(f'{properties.data_dir}{properties.main_data}{file_name}', index_col=0)
        msa_df = msa_df[msa_df.index != 'consensus']
        msa_df = msa_df[msa_df.index != 'reference']
        msa_df = msa_df.apply(convert, axis=1, result_type='broadcast')
        msa_df.to_csv(f'{properties.data_dir}{properties.processed_data}form_3_{gene}.csv')
