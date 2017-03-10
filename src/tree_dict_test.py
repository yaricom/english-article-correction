#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:03:04 2017

@author: yaric
"""

import unittest
import json

import tree_dict as td

class TestDeepTreeMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("../data/parse_train.txt") as f:
            data = json.load(f)
        tree_dict = data[1]
        root, index = td.treeFromJSON(tree_dict)
        cls.root = root

    def test_walk(self):
        nodes = [n for n in td.walk(self.root)]
        self.assertEqual(len(nodes), 125, "Nodes in the ROOT")
        
    def test_leaves(self):
        leaves = self.root.leaves()
        self.assertEqual(len(leaves), 43, "Leaves in the ROOT")
        
    def test_leaves_s_indexes(self):
        leaves = self.root.leaves()
        self.assertEqual(len(leaves), 43, "Leaves in the ROOT")
        
        index = 0
        for l in leaves:
            self.assertEqual(l.s_index, index, "Index of leaf")
            index += 1
        
    def test_subtrees(self):
        subtrees = self.root.subtrees()
        self.assertEqual(len(subtrees), 82, "Subtrees in the ROOT [min_childs = 1]")
        
        subtrees = self.root.subtrees(min_childs = 2)
        self.assertEqual(len(subtrees), 28, "Subtrees in the ROOT [min_childs = 2]")
    
    def test_np_subtrees(self):
        subtrees = self.root.subtrees()
        np_subtrees = 0
        for st in subtrees:
            if st.name == 'NP':
                np_subtrees += 1
                
        self.assertEqual(np_subtrees, 13, "NP Subtrees in the ROOT")
        
    def test_deepNPSubtrees(self):
        subtrees = self.root.deepNPSubtrees()
        self.assertEqual(len(subtrees), 11, "Deep NP Subtrees in the ROOT")
        
    def test_leaves_with_pos(self):
        leaves = self.root.leavesWithPOS('DT')
        self.assertEqual(len(leaves), 3, "Leaves with POS 'DT' in the ROOT")
        
    def test_dpaSubtrees(self):
        subtrees = self.root.dpaSubtrees()
        self.assertEqual(len(subtrees), 4, "DPA Subtrees in the ROOT")
 
class TestShallowTreeMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("../data/parse_train.txt") as f:
            data = json.load(f)
        tree_dict = data[723]
        root, index = td.treeFromJSON(tree_dict)
        cls.root = root
        
    def test_walk(self):
        nodes = [n for n in td.walk(self.root)]
        self.assertEqual(len(nodes), 33, "Nodes in the ROOT")
        
    def test_leaves(self):
        leaves = self.root.leaves()
        self.assertEqual(len(leaves), 14, "Leaves in the ROOT")
        
    def test_leaves_s_indexes(self):
        leaves = self.root.leaves()
        self.assertEqual(len(leaves), 14, "Leaves in the ROOT")
        
        index = 0
        for l in leaves:
            self.assertEqual(l.s_index, index, "Index of leaf")
            index += 1
        
    def test_subtrees(self):
        subtrees = self.root.subtrees()
        self.assertEqual(len(subtrees), 19, "Subtrees in the ROOT [min_childs = 1]")
        subtrees = self.root.subtrees(min_childs = 2)
        self.assertEqual(len(subtrees), 2, "Subtrees in the ROOT [min_childs = 2]")
    
    def test_np_subtrees(self):
        subtrees = self.root.subtrees()
        np_subtrees = 0
        for st in subtrees:
            if st.name == 'NP':
                np_subtrees += 1
                
        self.assertEqual(np_subtrees, 4, "NP Subtrees in the ROOT")
        
    def test_deepNPSubtrees(self):
        subtrees = self.root.deepNPSubtrees()
        self.assertEqual(len(subtrees), 4, "Deep NP Subtrees in the ROOT")
        
    def test_leaves_with_pos(self):
        leaves = self.root.leavesWithPOS('DT')
        self.assertEqual(len(leaves), 1, "Leaves with POS 'DT' in the ROOT")
        
    def test_dpaSubtrees(self):
        subtrees = self.root.dpaSubtrees()
        self.assertEqual(len(subtrees), 0, "DPA Subtrees in the ROOT")
    
if __name__ == '__main__':
    unittest.main()
    

    
        