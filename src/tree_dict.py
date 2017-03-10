#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The parse tree dictionary parser

@author: yaric
"""
import json

class SNode(object):
    """
    Represents specific node of parse tree 
    """
    
    def __init__(self, name, s_index = -1, pos = None):
        """
        Creates new node
        Arguments:
            name: the node name
            s_index: the index of unit in sentence for leaf nodes [optional]
            pos: the part of speech for leaf nodes [optional]
        """
        self.name = name
        self.s_index = s_index
        self.pos = pos
        self.children = list()
        
    def leaves(self):
        """
        Returns list of all leaves in this node
        """
        return [n for n in walk(self) if n.isLeaf()]
    
    def leavesWithPOS(self, pos):
        """
        Returns list of all tree leaves with specified POS
        """
        return [n for n in walk(self) if n.isLeaf() and n.pos == pos]
    
    def subtrees(self):
        """
        Returns all subtrees in this node
        """
        return [n for n in walk(self) if len(n.children) > 1]
    
    def dpSubtrees(self):
        """
        Returns all leaves containing exactly one determiner (DT) in form of article [a, an, the]
        """
        dp_trees = list()
        subtrees = self.subtrees()
        for st in subtrees:
            if st.name == 'NP':
                dt_leaves = st.leavesWithPOS('DT')
                if len(dt_leaves) == 1 and any(dt_leaves[0].name == name for name in ['a', 'an', 'the']):
                    dp_trees.append(st)
                
        return dp_trees
                
        
        
    def isLeaf(self):
        """
        Returns True if this node is leaf
        """
        return len(self.children) == 0
        

def walk(node):
    """ 
    Iterates tree node in pre-order depth-first search order 
    Argument:
        node: the tree node
    Return:
        the generator to iterate over tree node in pre-order depth-first search order 
    """
    yield node
    for child in node.children:
        for n in walk(child):
            yield n
            
def printNode(node):
    """
    Print treen node
    Arguments:
        node: the tree node
    """
    print(node.name + " | " + str(node.s_index) + " | " + str(node.pos))
            
def treeFromDict(d, s_index = 0, root = None):
    """
    Builds tree of SNode from provided dictionary
    Arguments:
        d: the dictionary with tree representation
        s_index: the sentence index of last leaf node
        root: the root node
    Return:
        the tuple with root node of the tree and the sentence index of last leaf node
    """
    if root == None:
        root = SNode(d["name"]) 
      
    children = d["children"]    
    for child in children:
        if len(child["children"]) == 0:
            # the leaf node found
            node = SNode(child["name"], s_index, root.name)
            root.children.append(node)
            s_index += 1
        else:
            # the interior node
            node = SNode(child["name"])
            _, s_index = treeFromDict(child, s_index, node)            
            root.children.append(node)
                
    return (root, s_index)

def treeFromList(l):
    """
    Builds tree of SNode from provided list
    Arguments:
        l: the list with tree representation
    Return:
        the tuple with root node of the tree and the sentence index of last leaf node
    """
    root = SNode("S")
    s_index = 0
    for child in l:
        node = SNode(child["name"])
        _, s_index = treeFromDict(child, s_index, node)
        root.children.append(node)
        
    return (root, s_index)
        

def treeFromJSON(json_str):
    """
    Builds tree from JSON string
    Arguments:
        json_str: the JSON string with tree data
    Return:
        the tree build from provided JSON structure
    """
    if json_str["name"] == 'TOP':
        return treeFromDict(json_str['children'][0])
    elif json_str["name"] == 'INC':
        return treeFromList(json_str['children'])
    else:
        raise("Unknown tree format found: " + json_str["name"])
    
    

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
    tree_str = json.dumps(data[1])
    print(tree_str)
    tree =  json.loads(tree_str)
    acc = treeStringFromDict(tree['children'][0])
    
    print("+++++++++++++++++++++ Tree string")
    print(acc)
    
    root, index = treeFromJSON(tree)
    tree_nodes = walk(root)
    print("+++++++++++++++++++++ Nodes")
    print("Last index: " + str(index))
    for n in tree_nodes:
        printNode(n)

    print("+++++++++++++++++++++ Leaves")
    leaves = root.leaves()
    for n in leaves:
        printNode(n)
        
    print("+++++++++++++++++++++ Subtrees")
    subtrees = root.subtrees()
    for st in subtrees:
        printNode(st)
        
    print("+++++++++++++++++++++ NP Subtrees")
    subtrees = root.subtrees()
    for st in subtrees:
        if st.name == 'NP':
            print("---")
            printNode(st)
            for l in st.leaves():
                printNode(l)
    

    print("+++++++++++++++++++++ Leaves with POS")
    leaves = root.leavesWithPOS('DT')
    for n in leaves:
        printNode(n)
            
    print("+++++++++++++++++++++ DP Subtrees")
    subtrees = root.dpSubtrees()
    for st in subtrees:
        print("---")
        for l in st.leaves():
            printNode(l)