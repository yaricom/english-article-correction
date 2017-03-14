#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:33:45 2017

@author: yaric
"""
import os
import unittest
import itertools
import json

import numpy as np

import tree_dict as td
import data_set as ds
import results_generator as rg
import config
import utils

class TestResultsGeneratorMethods(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.labels = np.array([[ 0.9       ,  0.1       ,  0.        ,  0.        ],
                               [ 0.55      ,  0.45      ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype = 'f')
        cls.text_data = json.loads('[["Silly", "libs", "--", "the", "government", "owns", "the", "uterus", "of", "all", "women", "."], \
                                ["It", "is", "in", "a", "Constitution", "!"]]')
        cls.parse_trees_list = json.loads('[{"name": "TOP", "children": [{"name": "S", "children": [{"name": "NP", "children": [{"name": "JJ", "children": [{"name": "Silly", "children": []}]}, {"name": "NNS", "children": [{"name": "libs", "children": []}]}]}, {"name": ":", "children": [{"name": "--", "children": []}]}, {"name": "NP", "children": [{"name": "DT", "children": [{"name": "the", "children": []}]}, {"name": "NN", "children": [{"name": "government", "children": []}]}]}, {"name": "VP", "children": [{"name": "VBZ", "children": [{"name": "owns", "children": []}]}, {"name": "NP", "children": [{"name": "NP", "children": [{"name": "DT", "children": [{"name": "the", "children": []}]}, {"name": "NN", "children": [{"name": "uterus", "children": []}]}]}, {"name": "PP", "children": [{"name": "IN", "children": [{"name": "of", "children": []}]}, {"name": "NP", "children": [{"name": "DT", "children": [{"name": "all", "children": []}]}, {"name": "NNS", "children": [{"name": "women", "children": []}]}]}]}]}]}, {"name": ".", "children": [{"name": ".", "children": []}]}]}]}, \
                                        {"name": "TOP", "children": [{"name": "S", "children": [{"name": "NP", "children": [{"name": "PRP", "children": [{"name": "It", "children": []}]}]}, {"name": "VP", "children": [{"name": "VBZ", "children": [{"name": "is", "children": []}]}, {"name": "PP", "children": [{"name": "IN", "children": [{"name": "in", "children": []}]}, {"name": "NP", "children": [{"name": "DT", "children": [{"name": "the", "children": []}]}, {"name": "NNP", "children": [{"name": "Constitution", "children": []}]}]}]}]}, {"name": ".", "children": [{"name": "!", "children": []}]}]}]}]')
        
        cls.test_list = [[None, None, None, [0, 0.9], None, None, [0, 0.55], None, None, None, None, None], 
                     [None, None, None, [3, 1.0], None, None]]
        
    def test_predictions_from_labels(self):
        test_list = itertools.chain(*self.test_list)
        out_list = rg.predictionsFromLabels(self.labels, self.text_data, self.parse_trees_list)
        out_list = itertools.chain(*out_list)
        
        # compare results with test
        for test, res in zip(test_list, out_list):
            if test != None:
                self.assertEqual(test[0], res[0], "Not equal class labels: %s != %s" % (test[0], res[0]))
                self.assertAlmostEqual(test[1], res[1], 5, "Not equal confidence values: %s != %s" % (test[1], res[1]))
            else:
                self.assertEqual(test, res, "Not equal items: %s != %s" % (test, res))
        
    
    def test_predictions_for_sentence(self):    
        test_list = itertools.chain(*self.test_list)

        out_list = list()
        l_index = 0    
        for i in range(len(self.parse_trees_list)):
            sentence = self.text_data[i]
            node, _ = td.treeFromDict(self.parse_trees_list[i]) # the parse tree for sentence
            dpa_nodes = node.dpaSubtrees()
            s_list = rg.predictionsForSentence(sentence, self.labels[l_index : l_index + len(dpa_nodes),], dpa_nodes)
            out_list.append(s_list)
            # move to the next sentence labels
            l_index += len(dpa_nodes)
    
        out_list = itertools.chain(*out_list)

        # compare results with test
        for test, res in zip(test_list, out_list):
            if test != None:
                self.assertEqual(test[0], res[0], "Not equal class labels: %s != %s" % (test[0], res[0]))
                self.assertAlmostEqual(test[1], res[1], 5, "Not equal confidence values: %s != %s" % (test[1], res[1]))
            else:
                self.assertEqual(test, res, "Not equal items: %s != %s" % (test, res))
    
    def test_predictionsFromTagLabels(self):
        test_list = itertools.chain(*self.test_list)
        out_list = rg.predictionsFromTagLabels(self.text_data, self.labels)
        out_list = itertools.chain(*out_list)
        
        # compare results with test
        for test, res in zip(test_list, out_list):
            if test != None:
                self.assertEqual(test[0], res[0], "Not equal class labels: %s != %s" % (test[0], res[0]))
                self.assertAlmostEqual(test[1], res[1], 5, "Not equal confidence values: %s != %s" % (test[1], res[1]))
            else:
                self.assertEqual(test, res, "Not equal items: %s != %s" % (test, res))
        
    
    def test_save_predictions(self):
        predictions = np.array(
                [
                        [[0,1], [0,1], [ds.DT.A, 0.6],[0,1], [0,1],[0,1], [0,1], [ds.DT.THE, 0.8]],
                        [[0,1], [ds.DT.AN, 0.6],[0,1], [0,1], [0,1], [ds.DT.A, 0.8],[0,1], [0,1]],
                        [[0,1], [0,1],[0,1], [0,1], [0,1], [ds.DT.A, 0.8],[0,1], [0,1]]
                ], dtype = "f")
        file = config.unit_tests_dir + "/save_predictions_test.txt"
        if os.path.exists(config.unit_tests_dir) == False:
            os.makedirs(config.unit_tests_dir)
        
        rg.savePredictions(predictions, file)
        
        # test saved predictions
        saved = utils.read_json(file)
        self.assertIsNotNone(saved, "Failed to save predictions")
        self.assertEqual(len(predictions), len(saved), "Wrong size of saved sentences")
        
        for i in range(len(predictions)):
            pr = predictions[i]
            s_pr = saved[i]
            self.assertEqual(len(pr), len(s_pr), 
                             "Wrong senetence length [%d] in saved predictions at index: %d" 
                             % (len(s_pr), i))
            for j in range(len(pr)):
                if pr[j][0] == 0:
                    self.assertIsNone(s_pr[j], 
                                      "Wrong saved None prediction class at: %d, %d" % (i, j))
                else:
                    art = ds.DT.nameByValue(pr[j][0]).lower()
                    self.assertEqual(art, s_pr[j][0],
                                     "Wrong article prediction class at: %d, %d" % (i, j))
                    self.assertAlmostEqual(pr[j][1], s_pr[j][1], places = 2,
                                     msg = "Wrong article confidence value at: %d, %d" % (i, j))
                    

if __name__ == '__main__':
    unittest.main()