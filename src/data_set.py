#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The data set loader. It will create numeric features matrix by chunking Noun 
Phrases (NP) from provided data corpus and substituting words in the NP with
corresponding indices from Glove vectors.

@author: yaric
"""

import json
import numpy as np

import tree_dict as td
from utils import POS


def __read_json(file):
    """
    Loads JSON data from file
    Arguments:
        file: the path to the JSON file
    Return:
        the dictionary with parsed JSON
    """
    with open(file) as f:
        data = json.load(f)
    return data

def extractFeatures(node, features, row, glove):
    """
    Method to extract features from provided node and store it in features array
    Arguments:
        node: the parse tree node with sentence
        features: the ndarray to store features row
        row: the row at features array to hold data
        glove: the glove indices map
    """
    """
    Features map:
    DT index | NN(S) index | NP length | 
    CC | CD | DT | EX | FW | IN | JJ | JJR | JJS| LS | MD | NN | NNS | NNP | NNPS | PDT	| POS | PRP | PRP$ | RB | RBR | RBS | RP | SYM
    TO | UH | VB | VBD | VBG | VBN | VBP | VBZ | WDT | WP | WP$ | WRB
    """
    # find all NPs 
    np_trees = list()
    subtrees = node.subtrees()
    for st in subtrees:
        if st.name == 'NP':
            
    

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
    corrections = __read_json(corrections_file)
    parse_trees_list = __read_json(parse_tree_file)
    #glove_indices = __read_json(glove_file)
    
    # The sanity checks
    if len(corrections) != len(parse_trees_list):
        raise("Corrections list lenght: %d, not equals parse trees count: %d" 
              % (len(corrections), len(parse_trees_list)))
    
    # iterate over constituency parse trees and extract features
    index = 0
    corrected_sentences = 0
    for tree_dict in parse_trees_list:
        # get corrections list for the sentence
        s_corr = corrections[index]
        # get parse tree for sentence
        tree, _ = td.treeFromDict(tree_dict)
        # do sanity check
        leaves = tree.leaves()
        if len(s_corr) != len(leaves):
            raise Exception("Corrections list lenght: %d not equal tree leaves count: %d at index: %d"  % (len(s_corr), len(leaves), index))
        # check if sentence has corrections
        hasCorrections = any(w != None for w in s_corr)
        if hasCorrections:
            corrected_sentences += 1
        
        index += 1
        
        
    print("Total senteces: %d, sentences with correction: %d" % (len(corrections), corrected_sentences))
        
        
if __name__ == '__main__':
    data_dir = "../data/"
    __create(data_dir + "sentence_train.txt", data_dir + "parse_train.txt",
             data_dir + "glove_train.txt", data_dir + "corrections_train.txt")
