#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:21:58 2017

@author: yaric
"""

import unittest

import ngram_res as nr

class TestNgramResultsMethods(unittest.TestCase):
    
    def test_buildPredictions(self):
        text_data =  [["Since", "you", "see", "that", "in", "most", "notionally", "ideological", "autocracies", ",", "whether", "\"", "religious", "\"", "or", ",", "e.g.", ",", "\"", "Communist", "\"", ",", "I", "do", "n\'t", "think", "its", "unreasonable", ",", "though", "Iran", "could", "be", "exceptional", "."], 
        ["Dice", ",", "funny", "that", "you", "redefine", "the", "govt", "as", "the", "Shia", "faction", ",", "then", "use", "your", "newly", "coined", "definition", "to", "change", "a", "meaning", "of", "my", "post", ",", "then", "call", "me", "\"", "deeply", "ignorant", "\"", "for", "saying", "something", "other", "than", "what", "I", "did", "."], 
        ["But", "those", "are", "a", "rules", "on", "a", "Lefty", "site", "where", "logic", "takes", "second", "place", "to", "name", "calling", "and", "sophistry", "."], 
        ["So", "then", "where", "is", "the", "civil", "war", "?"]]
        
        predictor = lambda x, context: 0.7 if x == 'a' else 0.6 if x == 'the' else 0.3 if x == 'an' else -1
                
        results = nr.buildPredictions(text_data, predictor)
        
        
        self.assertEqual(len(results), 4, "Wrong results list length")
        for s, r in zip(text_data, results):
            self.assertEqual(len(r), len(s), "Prediction results for sentence has wrong legth")
            for w, p in zip(s, r):
                if w in nr.dt_list:
                    if w != 'a':
                        self.assertEqual(p[1], 0.7, "Wrong confidence: " + str(p))
                else:
                    self.assertIsNone(p, "Prediction should be none for word: " + w)
        
        
if __name__ == '__main__':
    unittest.main()