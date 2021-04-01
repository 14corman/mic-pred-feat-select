# -*- coding: utf-8 -*-
"""
Created on Mar 24 8:02 PM 2021

@author: Cory Kromer-Edwards

This preprocessing file looks to create input form 1 where the list of annotations are created into a list
of scores using the following methods:
- comparing substitutions using BLOSUM-62 matrix
- Frame shifts                    = -10
- Inserts (no matter length)      = -5
- Deletions (no matter length)    = -5
- Duplications (no matter length) = -2
"""

import pandas as pd
from Bio.Align import substitution_matrices

blosum62 = substitution_matrices.load('BLOSUM62')


def calc_score(row):
    if row.Type == 'sub':
        return blosum62[row.Ref, row.AA]    # Matrix is square, so ref and aa are interchangable
    elif row.Type == 'fs':
        return -10
    elif row.Type == 'ins':
        return -5
    elif row.Type == 'del':
        return -5
    elif row.Type == 'dup':
        return -2


def create_form_one(csv_name):
    ann_df = pd.read_csv(f'{data_dir}{csv_name}')
    ann_df = ann_df.drop(['% coverage'], axis=1)
    ann_df['Score'] = ann_df.apply(calc_score, axis=1)
    ann_df = ann_df.drop(['Type', 'Ref', 'Position(s)', 'AA'], axis=1)

    # First, we want to groub by id and gene so we get all combined scores
    # Second, we want to get the made list of Score rows
    # Third, we want to make want to make each row into a list
    # Finally, we want to reset the index for each row so that we get individual rows per each element in the group
    # Found: https://stackoverflow.com/a/22221675
    ann_df.groupby(['Isolate_ID', 'Gene'])['Score'].apply(list).reset_index(name='ann_lists')

    ann_df.to_csv(f'{data_dir}processed/form_1.csv', index=False)
