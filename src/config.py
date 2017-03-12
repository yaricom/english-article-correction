#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holds common configuration parameters

@author: yaric
"""

# The root path to the data directory
data_dir = "../data"
# The output directory
out_dir = "../out"
# The intermediate output directory
intermediate_dir = out_dir + "/intermediate" 
# The directory to store unit test results
unit_tests_dir = intermediate_dir + "/u_tests"

#
# The train raw corpora
#
sentence_train_path = data_dir + "/sentence_train.txt"
parse_train_path = data_dir + "/parse_train.txt"
glove_train_path = data_dir + "/glove_train.txt"
corrections_train_path = data_dir + "/corrections_train.txt"

#
# The validate raw corpora
#
sentence_validate_path = data_dir + "/sentence_test.txt"
parse_validate_path = data_dir + "/parse_test.txt"
glove_validate_path = data_dir + "/glove_test.txt"
corrections_validate_path = data_dir + "/corrections_test.txt"

#
# The test raw corpora
#
sentence_test_path = data_dir + "/sentence_private_test.txt"
parse_test_path = data_dir + "/parse_private_test.txt"
glove_test_path = data_dir + "/glove_private_test.txt"
corrections_test_path = data_dir + "/corrections_private_test.txt"

#
# The train processed corpora
#
train_features_path = intermediate_dir + "/train_features.npy"
train_labels_path = intermediate_dir + "/train_labels.npy"

#
# The validate processed corpora
#
validate_features_path = intermediate_dir + "/validate_features.npy"
validate_labels_path = intermediate_dir + "/validate_labels.npy"

#
# The test processed corpora
#
test_features_path = intermediate_dir + "/test_features.npy"