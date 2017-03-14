#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The utility to convert prediction results from Numpy array of class labels with confidence
to list of lists where each child list corresponds to the text corpora sentence
with None at ordinary positions and list with class/confindece at predicted correction
of determinant (English article) wrong usage

@author: yaric
"""
import json
import argparse

import numpy as np

import tree_dict as td
import data_set as ds
import config
import utils

def saveSubmissionResults(f_type,
                          out_file = config.test_reults_path, 
                          labels_file = config.test_labels_prob_path, 
                          test_senetnces_file = config.sentence_test_path, 
                          test_parse_tree_file = config.parse_test_path,):
    """
    Saves submission results from provided prediction labels
    Argument:
        f_type: the type of features used for training
        out_file: the file to store predictions results in custom format
        labels_file: the path to the file with predcitions label Numpy array
        test_senetnces_file: the text's corpora file for test data
        test_parse_tree_file: the parse tree file for test data
    """
    labels = np.load(labels_file)
    text_data = utils.read_json(test_senetnces_file)
    parse_trees_list = utils.read_json(test_parse_tree_file)
    
    # generate predictions result
    if f_type == 'tree':
        predictions = predictionsFromLabels(labels, text_data, parse_trees_list)
    elif f_type == 'tags':
        predictions = predictionsFromTagLabels(text_data = text_data, labels = labels)
    else:
        raise Exception('Unknown features type: ' + f_type)
    print("Generated %d prediction sentences" % len(predictions))
    
    # sanity checks
    if len(predictions) != len(text_data):
        raise Exception("Number of sentences in predictions not equal to in test corpora: %d != %d" 
                        % (len(predictions), len(text_data)))
        
    for pr_s, text_s in zip(predictions, text_data):
        if len(pr_s) != len(text_s):
            raise Exception("Predictions sentence length not equal to text sentence length: %d != %d\nsentence: %s"
                            % (len(pr_s), len(text_s), text_s))
    
    # save to the file
    savePredictions(predictions, out_file)

def predictionsForSentence(sentence, labels, dpa_nodes):
    """
    Creates predictions list for specific sentence
    Arguments:
        sentence: the list of units in sentence
        labels: the labels list for given sentence
        dpa_nodes: the DT with English article nodes for given sentence
    Return:
        list of results in form [None, None, [class, confidence], None, ...]
    """
    # sanity check
    if len(labels) != len(dpa_nodes):
        raise Exception("labels lenght not equal to dpa nodes")
        
    dt_indices = list()
    for node in dpa_nodes:
        for dt_n in node.leavesWithPOS('DT'):
            dt_indices.append(dt_n.s_index)
            
    l_index = 0  
    res = list()      
    for i in range(len(sentence)):
        s_w = sentence[i]
        if i in dt_indices:
            # DP found
            if utils.dtIsArticle(s_w) == False:
                raise Exception("The determiner [%s] index at wrong position: [%d] for sentence: %s" % (s_w, i, sentence)) # sanity check
            max_lab_ind = np.argmax(labels[l_index]) # the most confident prediction
            res.append([max_lab_ind, labels[l_index, max_lab_ind]]) # [class, probability]
            l_index += 1 # move to the next prediction label
        else:
            # ordinary word
            res.append(None)
            
    # sanity check
    if l_index != len(labels):
        raise Exception("Not all labels was processed")
    
    return res        

def predictionsFromLabels(labels, text_data, parse_trees_list):
    """
    Create predictions results to be accepted by savePredictions based on features set
    generated with constituency parse trees
    Arguments:
        labels: the predicted labels from predictive model
        text_data: the text corpora as loaded from JSON
        parse_trees_list: the parse tree list as loaded from JSON for given text corpora
    Returns:
        list of predictions per sentence to be accepted by savePredictions
    """
    res_list = list()
    l_index = 0    
    for i in range(len(parse_trees_list)):
        sentence = text_data[i]
        node, _ = td.treeFromDict(parse_trees_list[i]) # the parse tree for sentence
        dpa_nodes = node.dpaSubtrees()
        s_list = predictionsForSentence(sentence, labels[l_index : l_index + len(dpa_nodes),], dpa_nodes)
        res_list.append(s_list)
        # move to the next sentence labels
        l_index += len(dpa_nodes)
        
    return res_list 

def predictionsFromTagLabels(text_data, labels):
    """
    Create predictions results to be accepted by savePredictions based on features set
    generated with pos tags
    Arguments:
        labels: the predicted labels from predictive model
        text_data: the text corpora as loaded from JSON
    Returns:
        list of predictions per sentence to be accepted by savePredictions
    """
    res_list = list()
    l_index = 0
    for s in text_data:
        row_data = list()
        for w in s:
            if w.lower() in ['a', 'an', 'the']:
                max_lab_ind = np.argmax(labels[l_index]) # the most confident prediction
                row_data.append([max_lab_ind, labels[l_index, max_lab_ind]]) # [class, probability]
                l_index += 1 # move to the next prediction label
            else:
                # ordinary word
                row_data.append(None)
        res_list.append(row_data)
        
    # sanity check
    if l_index != len(labels):
        raise Exception("Not all labels was processed")
        
    return res_list 
    
def savePredictions(predictions, file):
    """
    Method to save predictions results.
    Arguments:
        predictions: the list of lists with predictions per sentences for each unit
        [
             [None,[class, confidece],None],
             [None,None,[class, confidece]],
             ...
             [[class, confidece],None,None]
        ]
        file: the file path to save
    """
    out = list()
    for s in predictions:
        out_s = list()
        for w in s:
            if w == None:
                # None -> None
                out_s.append(None)
            elif w[0] < 0 or w[0] > ds.DT.THE.value:
                # sanity check
                raise Exception("Wrong class found: " + w[0])
            elif w[0] == 0:
                # no corrections was suggested
                out_s.append(None)
            else:
                # class -> article
                class_label = ds.DT.nameByValue(w[0]).lower()
                out_s.append([class_label, float(w[1])])
        # Append sentence
        out.append(out_s)
        
    # save result to JSON
    with open(file, mode = 'w') as f:
        json.dump(out, f)
    
    print("Prediction results saved to: " + file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The results generator transforming predicted labes array into submission compatible format')
    parser.add_argument('--out_file', default=config.test_reults_path, 
                        help='the file to store predictions results in custom format')
    parser.add_argument('--labels_file', default=config.test_labels_prob_path, 
                        help='the path to the file with predictions label Numpy array')
    parser.add_argument('--test_sentences_file', default=config.sentence_test_path, 
                        help="the text's corpora file for test data")
    parser.add_argument('--test_parse_tree_file', default=config.parse_test_path, 
                        help='the parse tree file for test data')
    parser.add_argument('--f_type', default='tree',
                        help='the type of features used for training [tree, tags]')
    args = parser.parse_args()
    
    print("Generating test results for features with type: %s" % args.f_type)
    
    saveSubmissionResults(f_type = args.f_type,
                          out_file = args.out_file, 
                          labels_file = args.labels_file, 
                          test_senetnces_file = args.test_sentences_file, 
                          test_parse_tree_file = args.test_parse_tree_file)
