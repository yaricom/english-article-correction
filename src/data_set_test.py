#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The test cases for data set implementation

@author: yaric
"""

import unittest
import numpy as np
import os
import json
import itertools

import utils
import config
import data_set as ds
import tree_dict as td

class TestDataSetMethods(unittest.TestCase):

    def test_extract_features(self):
        text_data = utils.read_json(config.corrections_train_path)
        corrections = utils.read_json(config.corrections_train_path)
        parse_trees_list = utils.read_json(config.parse_train_path)
        glove_indices = utils.read_json(config.glove_train_path)
        
        index = 1
        tree, _ = td.treeFromDict(parse_trees_list[index])
        
        features, labels = ds.extractFeatures(node = tree, 
                                              sentence = text_data[index],
                                              glove = glove_indices[index],
                                              corrections = corrections[index])
        
        self.assertEqual(len(features), 3, "Wrong features list size")
        self.assertEqual(features.shape[1], ds.n_features,
                          "Wrong feature dimensions: %d" % features.shape[1])
        
        # check labels
        labels_test = np.array([0, ds.DT.A, ds.DT.THE], dtype = "int")
        self.assertTrue(np.all(labels == labels_test), "Wrong labels generated")
        
        # check features
        features_test = np.zeros((3, ds.n_features), dtype = "f")
        features_test[0, 0] = 2
        features_test[0, 1] = 15189
        features_test[0, 2] = 1
        features_test[0, 3] = 0
        features_test[0, ds.offset + ds.POS.DT.value] = 1
        features_test[0, ds.offset + ds.POS.NN.value] = 1
        
        features_test[1, 0] = 2
        features_test[1, 1] = 19803
        features_test[1, 2] = 1
        features_test[1, 3] = 0
        features_test[1, ds.offset + ds.POS.DT.value] = 1
        features_test[1, ds.offset + ds.POS.NNP.value] = 1
        features_test[1, ds.offset + ds.POS.NN.value] = 1
        
        features_test[2, 0] = 6
        features_test[2, 1] = 2062
        features_test[2, 2] = 1
        features_test[2, 3] = 0
        features_test[2, ds.offset + ds.POS.DT.value] = 1
        features_test[2, ds.offset + ds.POS.NN.value] = 2
        features_test[2, ds.offset + ds.POS.IN.value] = 1
        features_test[2, ds.offset + ds.POS.PRP_.value] = 1
        
        select = features == features_test
        if np.all(select) == False:
            print(np.argmin(select, axis = 0))
            self.fail("Wrong features generated")

         
    def test_create_train_data_set(self):
        features, labels = ds.create(
                corpus_file = config.sentence_train_path, 
                parse_tree_file = config.parse_train_path,
                glove_file = config.glove_train_path, 
                corrections_file = config.corrections_train_path)
        
        self.assertEqual(len(features), len(labels), 
                          "The train features list has size not equal to the labels")
        self.assertEqual(features.shape[1], ds.n_features,
                          "Wrong feature dimensions: %d" % features.shape[1])
       
    def test_create_validate_data_set(self):
        features, labels = ds.create(
                corpus_file = config.sentence_validate_path, 
                parse_tree_file = config.parse_validate_path,
                glove_file = config.glove_validate_path, 
                corrections_file = config.corrections_validate_path)
        
        self.assertEqual(len(features), len(labels), 
                          "The validate features list has size not equal to the labels")
        self.assertEqual(features.shape[1], ds.n_features,
                          "Wrong feature dimensions: %d" % features.shape[1])
        
    def test_create_test_data_set(self):
        features, labels = ds.create(
                corpus_file = config.sentence_test_path, 
                parse_tree_file = config.parse_test_path,
                glove_file = config.glove_test_path, 
                corrections_file = config.corrections_test_path,
                test = True)
        
        self.assertIsNone(labels, "Labels should not be returned")
        self.assertGreater(features.shape[0], 0, "Empty features returned")
        self.assertEqual(features.shape[1], ds.n_features,
                          "Wrong feature dimensions: %d" % features.shape[1])

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
        
        ds.savePredictions(predictions, file)
        
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
                

    def test_predictions_for_sentence(self):    
        labels = np.array([[ 0.9       ,  0.1       ,  0.        ,  0.        ],
                           [ 0.55      ,  0.45      ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype = 'f')
    
        text_data = json.loads('[["Silly", "libs", "--", "the", "government", "owns", "the", "uterus", "of", "all", "women", "."], \
                                ["It", "is", "in", "a", "Constitution", "!"]]')
        parse_trees_list = json.loads('[{"name": "TOP", "children": [{"name": "S", "children": [{"name": "NP", "children": [{"name": "JJ", "children": [{"name": "Silly", "children": []}]}, {"name": "NNS", "children": [{"name": "libs", "children": []}]}]}, {"name": ":", "children": [{"name": "--", "children": []}]}, {"name": "NP", "children": [{"name": "DT", "children": [{"name": "the", "children": []}]}, {"name": "NN", "children": [{"name": "government", "children": []}]}]}, {"name": "VP", "children": [{"name": "VBZ", "children": [{"name": "owns", "children": []}]}, {"name": "NP", "children": [{"name": "NP", "children": [{"name": "DT", "children": [{"name": "the", "children": []}]}, {"name": "NN", "children": [{"name": "uterus", "children": []}]}]}, {"name": "PP", "children": [{"name": "IN", "children": [{"name": "of", "children": []}]}, {"name": "NP", "children": [{"name": "DT", "children": [{"name": "all", "children": []}]}, {"name": "NNS", "children": [{"name": "women", "children": []}]}]}]}]}]}, {"name": ".", "children": [{"name": ".", "children": []}]}]}]}, \
                                        {"name": "TOP", "children": [{"name": "S", "children": [{"name": "NP", "children": [{"name": "PRP", "children": [{"name": "It", "children": []}]}]}, {"name": "VP", "children": [{"name": "VBZ", "children": [{"name": "is", "children": []}]}, {"name": "PP", "children": [{"name": "IN", "children": [{"name": "in", "children": []}]}, {"name": "NP", "children": [{"name": "DT", "children": [{"name": "the", "children": []}]}, {"name": "NNP", "children": [{"name": "Constitution", "children": []}]}]}]}]}, {"name": ".", "children": [{"name": "!", "children": []}]}]}]}]')
        
        test_list = [[None, None, None, [0, 0.9], None, None, [0, 0.55], None, None, None, None, None], 
                     [None, None, None, [3, 1.0], None, None]]
        test_list = itertools.chain(*test_list)

        out_list = list()
        l_index = 0    
        for i in range(len(parse_trees_list)):
            sentence = text_data[i]
            node, _ = td.treeFromDict(parse_trees_list[i]) # the parse tree for sentence
            dpa_nodes = node.dpaSubtrees()
            s_list = ds.predictionsForSentence(sentence, labels[l_index : l_index + len(dpa_nodes),], dpa_nodes)
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

        
if __name__ == '__main__':
    unittest.main()