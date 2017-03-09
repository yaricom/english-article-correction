#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 22:13:59 2017

@author: yaric
"""
import json

def treeStringFromDict(d):
    """
    Builds tree definition string from dictionary
    Arguments:
        d: the dictionary
    Return: the string representation of tree
    """
    children = d["children"]
    acc = " "
    for child in children:
        if len(child["children"]) == 0:
            acc = acc + child["name"]
        else:
            acc_c = treeStringFromDict(child)
            acc = acc + "(" + child["name"] + acc_c + ")" 

    return acc
       
    
    
if __name__ == "__main__":
    with open("../data/parse_train.txt") as f:
        data = json.load(f)
    tree_str = json.dumps(data[0]['children'])
    print(tree_str)
    tree =  json.loads(tree_str)
    acc = treeFromDict(tree[0])
    
    print("+++++++++++++++++++++")
    print(acc)