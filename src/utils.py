#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 21:40:08 2017

@author: yaric
"""
import json

import pandas as pd
    
    
def read_json(file):
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

def checkDataCorporaSanity(corrections_file, corpus_file):
    """
    Method to  quick data corpus sanity check. It checks if there is no intersections
    between corrected acrticles and text's articles, i.e. test if we really have  
    corrected articles in text corpora present for training.
    """
    cor_df = pd.read_json(corrections_file, dtype="string")
    text_df = pd.read_json(corpus_file, dtype="string")
    
    # select all corrected articles
    cor_sel = cor_df.isnull() == False
    
    text_art = text_df[cor_sel]
    cor_art = cor_df[cor_sel]
    
    # find intersection between two
    intersection_df = text_art == cor_art
    
    intersection = intersection_df[intersection_df == True].sum().sum()
    
    print("The number of intersections: %d" % intersection)
    
        
if __name__ == '__main__':
    data_dir = "../data/"
    checkDataCorporaSanity(data_dir + "corrections_train.txt", data_dir + "sentence_train.txt")