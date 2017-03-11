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
import utils

# The Part-of-Speech enumerations
class POS(Enum):
    CC = 1
    CD = 2
    DT = 3
    EX = 4
    FW = 5
    IN = 6
    JJ = 7
    JJR = 8
    JJS = 9
    LS = 10
    MD = 11
    NN = 12
    NNS = 13
    NNP = 14
    NNPS = 15
    PDT = 16
    POS = 17
    PRP = 18
    PRP_ = 19
    RB = 20
    RBR = 21
    RBS = 22
    RP = 23
    SYM = 24
    TO = 25
    UH = 26
    VB = 27
    VBD = 28
    VBG = 29
    VBN = 36
    VBP = 37
    VBZ = 38
    WDT = 39
    WP = 40
    WP_ = 41
    WRB = 42


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
    #text_data = utils.read_json(corpus_file)
    corrections = utils.read_json(corrections_file)
    parse_trees_list = utils.read_json(parse_tree_file)
    #glove_indices = utils.read_json(glove_file)
    
    # The sanity checks
    if len(corrections) != len(parse_trees_list):
        raise Exception("Corrections list lenght: %d, not equals parse trees count: %d" 
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
