#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The test cases for data set implementation

@author: yaric
"""

import unittest
import numpy as np

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
        self.assertEqual(len(features), len(labels), 
                          "The features list has size not equal to the labels")
        print(labels)
        # check labels
        labels_test = np.array([[0], [ds.DT.A], [ds.DT.THE], [ds.DT.THE]], dtype = "int")
        self.assertTrue(np.all(labels == labels_test), "Wrong labels generated")
        
        print(features)
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
        
    def test_create_data_set(self):
        features, labels = ds.create(
                corpus_file = config.sentence_train_path, 
                parse_tree_file = config.parse_train_path,
                glove_file = config.glove_train_path, 
                corrections_file = config.corrections_train_path)
        
        self.assertEqual(len(features), len(labels), 
                          "The features list has size not equal to the labels")
        
        
        
if __name__ == '__main__':
    unittest.main()
        
        
        