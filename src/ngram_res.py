#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:38:47 2017

@author: yaric
"""

import pickle
import argparse
import json

import numpy as np
import config
import utils

from nltk_ngram import LidstoneNgramModel, NgramModelVocabulary, NgramCounter, MLENgramModel

n_gram_left = 3
dt_list = ['a', 'an', 'the']
confidence_threshold = 0#.1

def buildPredictions(text_data, predictor):
    res_list = list()
    for sentence in text_data:
        row_data = list()
        for i in range(len(sentence)):
            w = sentence[i].lower()
            if w in dt_list:
                # collect previous words
                if i >= n_gram_left:
                    context = sentence[i - n_gram_left:i]
                else:
                    context = sentence[:i]
                if len(context) > 0:
                    # do predictions
                    predictions = [predictor(dt, context) for dt in dt_list]
                    max_lab_ind = np.argmax(predictions) # the most confident prediction
                    max_dt = dt_list[max_lab_ind]
                    if max_dt != w and predictions[max_lab_ind] > confidence_threshold:
                       row_data.append([max_dt, predictions[max_lab_ind]]) 
                    else:
                      # predicted already present DT or too low confidence
                      row_data.append(None) 
                else:
                    # what happens when DT at the beginning of sentence?
                    predictions = None
                    row_data.append(None) 
            else:
                # ordinary word
                row_data.append(None)
                
        # store results
        res_list.append(row_data)
        
    return res_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The results generator based on traine n-gram model')
    parser.add_argument('--out_file', default=config.test_reults_path, 
                        help='the file to store predictions results in custom format')
    parser.add_argument('--test_sentences_file', default=config.sentence_test_path, 
                        help="the text's corpora file for test data")
    parser.add_argument('--model_file', default=config.ngram_model_path, 
                        help='the path to the file with predictions label Numpy array')
    
    args = parser.parse_args()
    
    print("Generating test results, model: [%s], text: [%s]" % (args.model_file, args.test_sentences_file))
    
    with open(args.model_file, 'rb') as f:
        counter = pickle.load(f)
        
    # The predictor
    predictor = lambda x, context: counter.ngrams[len(context) + 1][tuple(context)].freq(x)
    
    text_data = utils.read_json(args.test_sentences_file)
    result_list = buildPredictions(text_data, predictor)
    
    if len(result_list) != len(text_data):
        raise Exception("Text list size not equal to results list size, %d != %d" % (len(text_data), len(result_list)))
    
    # Save results
    with open(args.out_file, mode = 'w') as f:
        json.dump(result_list, f)
    
    print("Prediction results saved to: " + args.out_file)
    
    """
    words = ['tax', 'deductible', 'to']
    
    the_prob = counter.ngrams[4][('tax', 'deductible', 'to',)].freq('the')
    a_prob = counter.ngrams[4][('tax', 'deductible', 'to',)].freq('a')
    print("Probability: the = %f, a = %f" % (the_prob, a_prob))
    
    counter.ngrams[2][('leaving',)].freq('the')
    counter.ngrams[2][('leaving',)].freq('a')
    """
