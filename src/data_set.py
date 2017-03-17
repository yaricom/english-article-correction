#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The data set generatorion and results persistence routines. 
As generator it will create numeric features matrix by chunking Noun Phrases 
(NP) from provided data corpus and substituting words in the NP with
corresponding indices from Glove vectors.

As results persistence it will encode predictions as list of lists in format 
[[null, null, null, null, null, null, null, ["the", 0.552054389029137], null]],
where null means no correction at this unit and ["the", 0.552054389029137] means
correction suggested to "the' with confidence 0.552054389029137

@author: yaric
"""
from enum import IntEnum
import numpy as np
import os
import argparse

import tree_dict as td
import utils
import config

# The Part-of-Speech enumerations
class POS(IntEnum):
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
    VBN = 30
    VBP = 31
    VBZ = 32
    WDT = 33
    WP = 34
    WP_ = 35
    WRB = 36
    
    @classmethod
    def valueByName(cls, name):
        """
        Find enum value by the name
        Raise:
            exception in case if the name not found
        """
        return cls.__members__[name].value
    
    @classmethod
    def hasPOSName(cls, name):
        """
        Finds if provided name is known as POS
        """
        return name in cls.__members__

# The determinant article labels enumeration
class DT(IntEnum):
    A = 1
    AN = 2
    THE = 3
    
    @classmethod
    def valueByName(cls, name):
        """
        Find enum value by the name
        Raise:
            exception in case if the name not found
        """
        return cls.__members__[name.upper()].value
    
    @classmethod
    def nameByValue(cls, value):
        """
        Find enum name by the value
        """
        if value < cls.A or value > cls.THE:
            raise Exception("Wrong value for enumeration: " + value)
            
        return [member.name for _, member in cls.__members__.items() if member.value == value][0]

# The offset for POS features start    
offset = 2
# The number of features extracted
n_features = offset + len(POS.__members__) + 1

def extractFeatures(node, sentence, glove, corrections = None):
    """
    Method to extract features from provided node and store it in features array
    Arguments:
        node: the parse tree node with sentence
        sentence: the sentence corpora to extract features from
        glove: the glove indices map
        corrections: the list of corrections [optional] if building training data set
    Return:
        tuple with array of features for found determiner phrases with articles and
        labels or None
    """
    """
    Features map:
    DT glove index | NN(S) glove index | if DT at the sentence start |
    CC | CD | DT | EX | FW | IN | JJ | JJR | JJS| LS | MD | NN | NNS | NNP | NNPS | PDT	| POS | PRP | PRP$ | RB | RBR | RBS | RP | SYM |
    TO | UH | VB | VBD | VBG | VBN | VBP | VBZ | WDT | WP | WP$ | WRB
    """
    labels = None
    
    dpa_subtrees = node.dpaSubtrees()
    features = np.zeros((len(dpa_subtrees), n_features), dtype = 'f')
    if corrections != None:
        labels = np.zeros((len(dpa_subtrees),), dtype = 'int')
        
    # collect features
    row = 0
    for st in dpa_subtrees:
        nn_node = None
        for node in st.leaves():
            # collect POS type
            pos_name = node.pos.replace("$", "_")
            if POS.hasPOSName(pos_name):
                pos_index = POS.valueByName(pos_name)
                features[row, offset + pos_index] += 1
            
            if node.pos == 'DT':
                # found DT with article
                features[row, 0] = glove[node.s_index]
                # store flag to mark if DT at the start of sentence
                if node.s_index == 0:
                    features[row, 2] = 1
                # store correction label if appropriate
                if corrections != None and corrections[node.s_index] != None:
                    labels[row] = DT.valueByName(corrections[node.s_index])
                    
            elif nn_node == None and any(node.pos == pos for pos in ['NN', 'NNS', 'NNP', 'NNPS']):
                # found first (proper) noun
                features[row, 1] = glove[node.s_index]
                nn_node = node
                
        # increment row index
        row += 1
        
    return (features, labels)
 
# The number of fetures with NGram
n_features_pos_tags = 14#13#11

def extractPosTagsFeatures(sentence, pos_tags, glove, corrections = None, train_on_errors_only = True):
    """
    Extracts features for specified sentence
    Arguments:
        sentence: the sentence corpora to extract features from
        pos_tags: the part-of-speech tags for sentence
        glove: the glove indices map
        corrections: the list of corrections [optional] if building training data set
        train_on_errors_only: if true than features set generated for training based on error detection (only corrections considered for labels), 
                              otherwise features set generated for general correct DT articles detection (existing articles in sentences and 
                              corrections considered for labels)
    Return:
        tuple with array of features for found determiner phrases with articles and
        labels or None
    
    Features map:    
    preceding word | and POS| DT | following word | and POS | 
    second following word | and POS | head | head PoS | second preceding word | and POS |
    
    """
    articles = 0
    for w in sentence:
        if w.lower() in ['a', 'an', 'the']:
            articles += 1
    
    features = np.zeros((articles, n_features_pos_tags), dtype = 'f')
    if corrections != None:
        labels = np.zeros((articles,), dtype = 'int')
    else:
        labels = None
        
    prev_w = np.zeros((6,), dtype = 'int')
    row = -1
    dta_s_index = -1
    next_word_ind = [-1, -1, -1]
    for i in range(len(sentence)):
        p_tos = pos_tags[i]
        w_g = glove[i]
        cor = corrections[i] if corrections != None else None
        w = sentence[i].lower()
        if w in ['a', 'an', 'the']:
            # collect features
            dta_s_index = i
            row += 1
            
            # store allegedly incorrect DT article index to features
            features[row, 2] = w_g
                        
            if train_on_errors_only == False:
                # store sentence DT article class in labels (it may be overwritten by correction later)
                labels[row] = DT.valueByName(w)
                
            if prev_w[1] > 0:
                # add previous if its set
                features[row, 0] = prev_w[0] # preceding word index
                features[row, 1] = prev_w[1] # preceding word TOS
                
            if prev_w[3] > 0:
                # add previous - 1 if its set
                features[row, 9] = prev_w[2] # preceding - 1 word index
                features[row, 10] = prev_w[3] # preceding - 1 word TOS
  
            if prev_w[5] > 0 and prev_w[5] in [POS.VB, POS.VBD]:
                # add previous - 1 if its set
                features[row, 11] = prev_w[4] # preceding - 2 word index
                features[row, 12] = prev_w[5] # preceding - 2 word TOS

            # find index of next feature word
            next_word_ind[0] = dta_s_index + 1
                         
            # store collect label if any (possibly overwrite incorrect one set previously from sentence)
            if cor != None:
                labels[row] = DT.valueByName(cor)
                
        elif i == next_word_ind[0]:
            if POS.hasPOSName(p_tos):
                features[row, 3] = w_g # following word index
                features[row, 4] = POS.valueByName(p_tos) # following word TOS
                next_word_ind[1] = i + 1
                # vowel
                if w[0] in 'aeiou':
                    features[row, 13] = 1
            else:
                # not a tagged POS found
                next_word_ind[0] += 1
        elif i == next_word_ind[1]:
            if POS.hasPOSName(p_tos):
                features[row, 5] = w_g # following word + 1 index
                features[row, 6] = POS.valueByName(p_tos) # following word + 1 TOS
            else:
                # not a tagged POS found
                next_word_ind[1] += 1
            
        if dta_s_index > 0 and features[row, 8] == 0 and p_tos in ['NN', 'NNS']:
            # store first head noun following DT
            features[row, 7] = w_g # head word index
            features[row, 8] = POS.valueByName(p_tos) # head PoS
            #if p_tos == 'NN':
            #    features[row, 14] = 1
            #elif p_tos == 'NNS':
            #    features[row, 14] = 2
            
        
        # store preceding
        if prev_w[3] > 0:
            prev_w[4] = prev_w[2]
            prev_w[5] = prev_w[3]
            
        if prev_w[1] > 0:
            prev_w[2] = prev_w[0]
            prev_w[3] = prev_w[1]
            
        if POS.hasPOSName(p_tos):
            prev_w[0] = w_g
            prev_w[1] = POS.valueByName(p_tos)
        
       
    return (features, labels)
    
           
def create(corpus_file, parse_tree_file, glove_file, corrections_file, test = False):
    """
    Creates new data set from provided files using NP acquired from constituency parse trees
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
    text_data = utils.read_json(corpus_file)
    parse_trees_list = utils.read_json(parse_tree_file)
    glove_indices = utils.read_json(glove_file)
    
    if test == False:
        corrections = utils.read_json(corrections_file)
    
    # The sanity checks
    #
    if len(text_data) != len(parse_trees_list):
        raise Exception("Text data corpora lenght: %d, not equals to the parse trees count: %d" 
                        % (len(text_data), len(parse_trees_list)))
    if test == False and len(corrections) != len(parse_trees_list):
        raise Exception("Corrections list lenght: %d, not equals to the parse trees count: %d" 
              % (len(corrections), len(parse_trees_list)))
    if len(glove_indices) != len(parse_trees_list):
        raise Exception("Glove indices list lenght: %d, not equals to the parse trees count: %d" 
              % (len(corrections), len(parse_trees_list)))
    
    # iterate over constituency parse trees and extract features
    features = None
    labels = None
    index = 0
    corrected_sentences = 0
    for tree_dict in parse_trees_list:
        # get corrections list for the sentence
        if test == False:
            s_corr = corrections[index]
        else:
            s_corr = None
        # get glove indices list for the sentence
        g_list = glove_indices[index]
        # get text corpora list for sentence
        t_list = text_data[index]
        # get parse tree for sentence
        tree, _ = td.treeFromDict(tree_dict)

        # do sanity checks
        #
        leaves = tree.leaves()
        if test == False and len(s_corr) != len(leaves):
            raise Exception("Corrections list lenght: %d not equal the tree leaves count: %d at index: %d" 
                            % (len(s_corr), len(leaves), index))
        if len(g_list) != len(leaves):
            raise Exception("Glove indices list lenght: %d not equal the tree leaves count: %d at index: %d" 
                            % (len(g_list), len(leaves), index))
        if len(t_list) != len(leaves):
            raise Exception("Text corpora list lenght: %d not equal the tree leaves count: %d at index: %d" 
                            % (len(t_list), len(leaves), index))
        
        # check if sentence has corrections
        #
        if test == False and any(w != None for w in s_corr):
            corrected_sentences += 1
            
        # generate features and labels
        #
        f, l = extractFeatures(node = tree, sentence = t_list, glove = g_list, corrections = s_corr)
        if index == 0:
            features = f
        else:
            features = np.concatenate((features, f), axis=0)
        
        if index == 0:
            labels = l
        elif test == False:
            labels = np.concatenate((labels, l))
        
        index += 1
        
        
    print("Features collected: %d" % (len(features)))
    
    return (features, labels)

def createWithPosTags(corpus_file, pos_tags_file, glove_file, corrections_file, test = False):
    """
    Creates new features set from provided files using pos tags
    Arguments:
        corpus_file: the file text corpus
        pos_tags_file: the file with pos tags for data corpus
        glove_file: the file with GloVe vectors indexes for data corpus
        corrections_file: the file with labeled corrections
        test: the flag to indicate whether test data set should be constructed
    Return:
        (features, labels): the tuple with features and labels. If test parameter is True then labels
        wil be None
    """
    text_data = utils.read_json(corpus_file)
    pos_tags = utils.read_json(pos_tags_file)
    glove_indices = utils.read_json(glove_file)
    
    if test == False:
        corrections = utils.read_json(corrections_file)
    
    # The sanity checks
    #
    if len(text_data) != len(pos_tags):
        raise Exception("Text data corpora lenght: %d, not equals to the pos tags lists length: %d" 
                        % (len(text_data), len(pos_tags)))
    if test == False and len(corrections) != len(pos_tags):
        raise Exception("Corrections list lenght: %d, not equals to the pos tags lists length: %d" 
              % (len(corrections), len(pos_tags)))
    if len(glove_indices) != len(pos_tags):
        raise Exception("Glove indices list lenght: %d, not equals to the pos tags lists length: %d" 
              % (len(corrections), len(pos_tags)))
    
    features = None
    labels = None
    index = 0
    for index in range(len(pos_tags)):
        # get corrections list for the sentence
        if test == False:
            s_corr = corrections[index]
        else:
            s_corr = None
        # get glove indices list for the sentence
        g_list = glove_indices[index]
        # get text corpora list for sentence
        t_list = text_data[index]
        # the pos tags for sentence
        pt_list = pos_tags[index]
        
        # do sanity checks
        #
        if test == False and len(s_corr) != len(pt_list):
            raise Exception("Corrections list lenght: %d not equal the pos tags count: %d at index: %d" 
                            % (len(s_corr), len(pt_list), index))
        if len(g_list) != len(pt_list):
            raise Exception("Glove indices list lenght: %d not equal the pos tags count: %d at index: %d" 
                            % (len(g_list), len(pt_list), index))
        if len(t_list) != len(pt_list):
            raise Exception("Text corpora list lenght: %d not equal the pos tags count: %d at index: %d" 
                            % (len(t_list), len(pt_list), index))
            
        # generate features and labels
        #
        f, l = extractPosTagsFeatures(pos_tags = pt_list, sentence = t_list, glove = g_list, 
                                      corrections = s_corr, train_on_errors_only = True)
        if index == 0:
            features = f
        else:
            features = np.concatenate((features, f), axis=0)
        
        if index == 0:
            labels = l
        elif test == False:
            labels = np.concatenate((labels, l))
    
    print("Features collected: %d with dimension: %d" % (features.shape[0], features.shape[1]))
        
    return (features, labels)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('corpora', help = 'the name of data coprora to process')
    parser.add_argument('--f_type', default = 'tree', 
                        help = 'the type of features to be generated [tree, tags]')
    args = parser.parse_args()
    
    
    if args.f_type in ['tree', 'tags'] == False:
        raise Exception("Unknown features type: " + args.f_type)
    
    # Create output directory
    if os.path.exists(config.intermediate_dir) == False:
        os.makedirs(config.intermediate_dir)
        
    print("Making %s features with type [%s]" % (args.corpora, args.f_type))
    
    # Create train data corpus
    #
    if args.corpora == 'train':
        if args.f_type == 'tree':
            features, labels = create(corpus_file = config.sentence_train_path, 
                                 parse_tree_file = config.parse_train_path,
                                 glove_file = config.glove_train_path, 
                                 corrections_file = config.corrections_train_path)
        elif args.f_type == 'tags':
            features, labels = createWithPosTags(corpus_file = config.sentence_train_path, 
                                 pos_tags_file = config.pos_tags_train_path,
                                 glove_file = config.glove_train_path, 
                                 corrections_file = config.corrections_train_path)
            
        np.save(config.train_features_path, features)
        np.save(config.train_labels_path, labels)
    
    # Create validate data corpus
    #
    elif args.corpora == 'validate':
        if args.f_type == 'tree':
            features, labels = create(corpus_file = config.sentence_validate_path, 
                                 parse_tree_file = config.parse_validate_path,
                                 glove_file = config.glove_validate_path, 
                                 corrections_file = config.corrections_validate_path)
        elif args.f_type == 'tags':
            features, labels = createWithPosTags(corpus_file = config.sentence_validate_path, 
                                 pos_tags_file = config.pos_tags_validate_path,
                                 glove_file = config.glove_validate_path, 
                                 corrections_file = config.corrections_validate_path)
            
        np.save(config.validate_features_path, features)
        np.save(config.validate_labels_path, labels)
    
    # Create test data features
    #
    elif args.corpora == 'test':
        if args.f_type == 'tree':
            features, _ = create(corpus_file = config.sentence_test_path, 
                             parse_tree_file = config.parse_test_path,
                             glove_file = config.glove_test_path, 
                             corrections_file = config.corrections_test_path,
                             test = True)
        elif args.f_type == 'tags':
            features, _ = create(corpus_file = config.sentence_test_path, 
                             pos_tags_file = config.pos_tags_test_path,
                             glove_file = config.glove_test_path, 
                             corrections_file = None,
                             test = True)
            
        np.save(config.test_features_path, features)
    else:
        raise Exception("Unknown coprpora type: " + args.corpora)
