#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 21:40:08 2017

@author: yaric
"""
import json
import pandas as pd
    
import config

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

def checkDataCorporaSanity(data_dir, corpora_name):
    """
    Method to  quick data corpus sanity check. It checks if there is no intersections
    between corrected acrticles and text's articles, i.e. test if we really have  
    corrected articles in text corpora present for training.
    Arguments:
        data_dir: the directory to look for corpora data
        corpora_name: the name of corpora to test
    """
    corrections_file = "%s/corrections_%s.txt" % (data_dir, corpora_name)
    corpus_file = "%s/sentence_%s.txt" % (data_dir, corpora_name)
    cor_df = pd.read_json(corrections_file, dtype="string")
    text_df = pd.read_json(corpus_file, dtype="string")
    
    # select all corrected articles
    cor_sel = cor_df.isnull() == False
    
    text_art = text_df[cor_sel]
    cor_art = cor_df[cor_sel]
    
    # find intersection between two
    intersection_df = text_art == cor_art
    
    intersection = intersection_df[intersection_df == True].sum().sum()
    
    print("The number of intersections: %d in corpora: %s" 
          % (intersection, corpora_name))
    if intersection > 0:
        raise Exception("The text and corrections has the same Articles at the same position")
    
        
if __name__ == '__main__':
    checkDataCorporaSanity(config.data_dir, "train")
    checkDataCorporaSanity(config.data_dir, "test")