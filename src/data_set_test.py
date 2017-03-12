#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The test cases for data set implementation

@author: yaric
"""

import unittest
import numpy as np
import os

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
                                              text = text_data,
                                              glove = glove_indices[index],
                                              corrections = corrections[index])
        
        self.assertEqual(len(features), 4, "Wrong features list size")
        self.assertEqual(features.shape[1], ds.n_features,
                          "Wrong feature dimensions: %d" % features.shape[1])
        
        # check labels
        labels_test = np.array([0, ds.DT.A, ds.DT.THE, ds.DT.THE], dtype = "int")
        self.assertTrue(np.all(labels == labels_test), "Wrong labels generated")
        
        # check features
        offset = 2
        features_test = np.zeros((4, ds.n_features), dtype = "f")
        features_test[0, 0] = 2
        features_test[0, 1] = 15189
        features_test[0, 2] = 1
        features_test[0, offset + ds.POS.DT.value] = 1
        features_test[0, offset + ds.POS.NN.value] = 1
        
        features_test[1, 0] = 2
        features_test[1, 1] = 19803
        features_test[1, 2] = 1
        features_test[1, offset + ds.POS.DT.value] = 1
        features_test[1, offset + ds.POS.NNP.value] = 1
        features_test[1, offset + ds.POS.NN.value] = 1
        
        features_test[2, 0] = 6
        features_test[2, 1] = 2062
        features_test[2, 2] = 1
        features_test[2, offset + ds.POS.DT.value] = 1
        features_test[2, offset + ds.POS.NN.value] = 2
        features_test[2, offset + ds.POS.IN.value] = 1
        features_test[2, offset + ds.POS.PRP_.value] = 1
        
        features_test[3, 0] = 6
        features_test[3, 1] = 2062
        features_test[3, 2] = 1
        features_test[3, offset + ds.POS.DT.value] = 1
        features_test[3, offset + ds.POS.NN.value] = 1

        self.assertTrue(np.all(features == features_test), "Wrong features generated")
        
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
                
    
        
if __name__ == '__main__':
    unittest.main()