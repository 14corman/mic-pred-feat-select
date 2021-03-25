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
"""
