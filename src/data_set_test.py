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
        text_data = utils.read_json(config.sentence_train_path)
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
        features_test[0, 2] = 0
        features_test[0, ds.offset + ds.POS.DT.value] = 1
        features_test[0, ds.offset + ds.POS.NN.value] = 1
        
        features_test[1, 0] = 2
        features_test[1, 1] = 19803
        features_test[1, 2] = 0
        features_test[1, ds.offset + ds.POS.DT.value] = 1
        features_test[1, ds.offset + ds.POS.NNP.value] = 1
        features_test[1, ds.offset + ds.POS.NN.value] = 1
        
        features_test[2, 0] = 6
        features_test[2, 1] = 2062
        features_test[2, 2] = 0
        features_test[2, ds.offset + ds.POS.DT.value] = 1
        features_test[2, ds.offset + ds.POS.NN.value] = 2
        features_test[2, ds.offset + ds.POS.IN.value] = 1
        features_test[2, ds.offset + ds.POS.PRP_.value] = 1
        
        select = features == features_test
        if np.all(select) == False:
            print(np.argmin(select, axis = 0))
            self.fail("Wrong features generated")
    
    def test_extract_Pos_Tags_Features(self):
        text_data = utils.read_json(config.sentence_train_path)
        corrections = utils.read_json(config.corrections_train_path)
        glove_indices = utils.read_json(config.glove_train_path)
        pos_tags = utils.read_json(config.pos_tags_train_path)
        
        
        s_index = 1
        
        features, labels = ds.extractPosTagsFeatures(text_data[s_index], pos_tags[s_index], 
                                                     glove_indices[s_index], corrections[s_index])
        self.assertEqual(len(features), 3, "Wrong features list size")
        self.assertEqual(features.shape[1], ds.n_features_pos_tags,
                          "Wrong feature dimensions: %d" % features.shape[1])
        
        # check labels
        labels_test = np.array([0, ds.DT.A, ds.DT.THE], dtype = "int")
        self.assertTrue(np.all(labels == labels_test), "Wrong labels generated")
        
        # check features
        """
        PrW | PrW POS | DTa | FlW | FlW POS | FlW2 | FlW2 POS | FlNNs (i > 0) | FlNNs (i > 0) POS |
        PrW2 | PrW2 POS | PrW3 (VB, VBD) | PrW3 (VB, VBD) POS | Vowel ([0, 1] | FlW (VB,VBD,VBG,VBN,VBP,VBZ) | FlW (VB,VBD,VBG,VBN,VBP,VBZ) POS
        """
        features_test = np.array([
                [21176.0, 27.0, 2.0, 15189.0, 12.0, 28.0, 6.0, 15189.0, 12.0, 18.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                [28.0, 6.0, 2.0, 19803.0, 14.0, 16560.0, 12.0, 16560.0, 12.0, 15189.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                [365.0, 27.0, 6.0, 2062.0, 12.0, 5.0, 6.0, 2062.0, 12.0, 4.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype = "f")
        
        select = features == features_test
        if np.all(select) == False:
            print(np.argmin(select, axis = 0))
            self.fail("Wrong features generated")
        
    def test_create_train_data_set_WithPosTags(self):
        print("Train -- ")
        features, labels = ds.createWithPosTags(
                 corpus_file = config.sentence_train_path, 
                 pos_tags_file = config.pos_tags_train_path,
                 glove_file = config.glove_train_path, 
                 corrections_file = config.corrections_train_path)
        
        self.assertEqual(len(features), len(labels), 
                          "The train features list has size not equal to the labels")
        self.assertEqual(features.shape[1], ds.n_features_pos_tags,
                          "Wrong train feature dimensions: %d" % features.shape[1])
        
    def test_create_validate_data_set_WithPosTags(self):
        print("Validate -- ")
        features, labels = ds.createWithPosTags(
                 corpus_file = config.sentence_validate_path, 
                 pos_tags_file = config.pos_tags_validate_path,
                 glove_file = config.glove_validate_path, 
                 corrections_file = config.corrections_validate_path)
        
        self.assertEqual(len(features), len(labels), 
                          "The validate features list has size not equal to the labels")
        self.assertEqual(features.shape[1], ds.n_features_pos_tags,
                          "Wrong validate feature dimensions: %d" % features.shape[1])
        
    def test_create_test_data_set_WithPosTags(self):
        print("Test -- ")
        features, labels = ds.createWithPosTags(
                 corpus_file = config.sentence_test_path, 
                 pos_tags_file = config.pos_tags_test_path,
                 glove_file = config.glove_test_path, 
                 corrections_file = None,
                 test = True)
        
        self.assertIsNone(labels, "Labels should not be returned")
        self.assertGreater(features.shape[0], 0, "Empty test features returned")
        self.assertEqual(features.shape[1], ds.n_features_pos_tags,
                          "Wrong test feature dimensions: %d" % features.shape[1])

    def test_create_train_data_set(self):
        print("Train - ")
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
        print("Validate - ")
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
        print("Test - ")
        features, labels = ds.create(
                corpus_file = config.sentence_test_path, 
                parse_tree_file = config.parse_test_path,
                glove_file = config.glove_test_path, 
                corrections_file = None,
                test = True)
        
        self.assertIsNone(labels, "Labels should not be returned")
        self.assertGreater(features.shape[0], 0, "Empty features returned")
        self.assertEqual(features.shape[1], ds.n_features,
                          "Wrong feature dimensions: %d" % features.shape[1])
    
if __name__ == '__main__':
    unittest.main()