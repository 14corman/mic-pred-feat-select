# -*- coding: utf-8 -*-
"""
Created on Aug 11 7:24 AM 2021

@author: Cory Kromer-Edwards

Main file to call when testing new data against NN model. Will run a test pipeline.
"""

import get_format_data
import nn
import nn_analysis

if __name__ == "__main__":
    print("Collecting and formatting data...")
    get_format_data.collect_data(1)
    
    print("Generating NN test file...")
    nn.generate_nn_files()

    print("Testing NN to get F1 score...")
    nn.test_nn_model()

    print("Running analysis against NN model...")
    nn_analysis.create_mic_heatmap()
    nn_analysis.create_error_report()

    print("Testing pipeline complete!")
