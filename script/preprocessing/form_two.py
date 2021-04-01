# -*- coding: utf-8 -*-
"""
Created on Mar 24 8:03 PM 2021

@author: Cory Kromer-Edwards

This preprocessing file looks to create input form 2 where the annotated Amino Acid
sequences will be converted into either -1 or 1 given the following criteria:
-1 if:
    1. The Amino Acid had no mutation
1 if:
    1. A SNP occurred at the position
    2. An insertion occurred (no matter how long the inserted sequence was, only a single -1 is given)
    2. A deletion occurred (no matter how long the deleted sequence was, only a single -1 is given)

Duplications are considered (-1) for the unmuttated Amino Acid and then the rest of the dupplication
    is treated as a single insert (1).
Frameshifts are considered a 1 for every position beyond where the frameshift occurred.
"""

import pandas as pd


def create_ones_list(row, reference=[]):
    # The first element of the row is the isolate id, so we must remove that first
    row = row[1:]

    # Since both sequences were part of a MSA, we can assume the lists are the same length.
    frameshift = False
    insert = False
    delete = False
    ones = []
    for p, r in zip(row, reference):

        # Frameshift occurred in the sequence
        if frameshift:
            ones.append(1)
            continue
        elif p == '?':
            frameshift = True
            ones.append(1)
            continue

        # Insert occurred in the sequence meaning reference has a -
        if insert and r == '-':
            continue
        elif r == '-':
            insert = True
            ones.append(1)
            continue
        else:
            insert = False

        # Delete occurred in the sequence meaning sequence has a -
        if delete and p == '-':
            continue
        elif p == '-':
            delete = True
            ones.append(1)
            continue
        else:
            delete = False

        # Either a substitution occurred or the 2 sequences are the same
        if p != r:
            ones.append(1)
        else:
            ones.append(-1)

    return ones


def create_form_two(csv_name):
    msa_df = pd.read_csv(f'{data_dir}{csv_name}')
    msa_df = msa_df[msa_df.Name != 'consensus']     # Remove consensus row

    # Get the reference row, turn it into a list of lists (there will only be 1 row, so 1 list)
    # Then, remove the first element which is the name of the row 'reference'
    reference = msa_df[msa_df.Name == 'reference'].values.tolist()[0][1:]

    # Remove reference row
    msa_df = msa_df[msa_df.Name != 'reference']

    # Create ones list for each row
    msa_df['ones'] = msa_df.apply(create_ones_list, axis=1, reference=reference)

    # Remove all gene columns
    msa_df = msa_df.drop(msa_df.columns[[x not in ['Name', 'ones'] for x in msa_df.columns]], axis=1)
    msa_df.to_csv(f'{data_dir}processed/form_2.csv', index=False)
