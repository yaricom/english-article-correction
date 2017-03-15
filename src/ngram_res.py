#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:38:47 2017

@author: yaric
"""

import pickle


from nltk_ngram import LidstoneNgramModel, NgramModelVocabulary, NgramCounter, MLENgramModel

with open("../out/counter.pkl", 'rb') as f:
    counter = pickle.load(f)
    
words = ['tax', 'deductible', 'to']

the_prob = counter.ngrams[4][('tax', 'deductible', 'to',)].freq('the')
a_prob = counter.ngrams[4][('tax', 'deductible', 'to',)].freq('a')
print("Probability: the = %f, a = %f" % (the_prob, a_prob))

counter.ngrams[2][('leaving',)].freq('the')
counter.ngrams[2][('leaving',)].freq('a')
