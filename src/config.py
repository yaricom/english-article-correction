#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holds common configuration parameters

@author: yaric
"""

# The root path to the data directory
data_dir = "../data"

#
# Train corpora
#
# The text corpora
sentence_train_path = data_dir + "/" + "sentence_train.txt"
# The corrections 
parse_train_path = data_dir + "/" + "parse_train.txt"
# The Glove vectors
glove_train_path = data_dir + "/" + "glove_train.txt"
# The corrections
corrections_train_path = data_dir + "/" + "corrections_train.txt"