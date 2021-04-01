# -*- coding: utf-8 -*-
"""
Created on Mar 24 6:48 PM 2021

@author: Cory Kromer-Edwards

Main file. Code execution starts here.
"""

import properties
import os

if __name__ == '__main__':
    properties = properties.load_properties(os.join(os.pwd(), "config.propertes"))
