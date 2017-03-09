#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The data set loader. It will create numeric features matrix by chunking Noun 
Phrases (NP) from provided data corpus and substituting words in the NP with
corresponding indices from Glove vectors. The feature spaces we'll try to use
as described in: ftp://ling.upenn.edu/papers/students/nrh/han-chodorow04.pdf

@author: yaric
"""

import json
import numpy as np

def __read_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

def __create(corpus_file, parse_tree_file, glove_file, corrections_file, test = False):
    """
    Creates new data set from provided files
    Arguments:
        corpus_file: the file text corpus
        parse_tree_file: the file with constituency parse trees build over data corpus
        glove_file: the file with GloVe vectors indexes for data corpus
        corrections_file: the file with labeled corrections
        test: the flag to indicate whether test data set should be constructed
    Return:
        (features, labels): the tuple with features and labels. If test parameter is True then labels
        wil be None
    """
    #text_data = __read_json(corpus_file)
    #corrections = __read_json(corrections_file)
    parse_tree_list = __read_json(parse_tree_file)
    #glove_indices = __read_json(glove_file)
    
    # iterate over constituency parse trees and extract NP
    index = 0
    for subtree in parse_tree_list:
        print(subtree)
        
        
if __name__ == '__main__':
    data_dir = "../data/"
    __create(data_dir + "sentence_train.txt", data_dir + "parse_train.txt",
             data_dir + "glove_train.txt", data_dir + "corrections_train.txt")
