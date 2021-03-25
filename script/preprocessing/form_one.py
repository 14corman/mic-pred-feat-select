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
"""
