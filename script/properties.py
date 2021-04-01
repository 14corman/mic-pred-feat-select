# -*- coding: utf-8 -*-
"""
Created on Mar 31 9:06 PM 2021

@author: Cory Kromer-Edwards

This file is made to load a properties file on code startup and have
 a namespace returned of loaded properties
"""
from argparse import Namespace


# Code taken from StackOverflow answer: https://stackoverflow.com/a/31852401
def load_properties(filepath, sep='=', comment_char='#'):
    """
    Read the file passed as parameter as a properties file.
    """
    props = {}
    with open(filepath, "rt") as f:
        for line in f:
            l = line.strip()
            if l and not l.startswith(comment_char):
                key_value = l.split(sep)
                key = key_value[0].strip()
                value = sep.join(key_value[1:]).strip().strip('"')
                props[key] = value
    return Namespace(**props)
