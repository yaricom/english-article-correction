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
        
    def __str__(self):
        """
        Returns string representation of this node.
        """
        return self.name + " | " + str(self.s_index) + " | " + str(self.pos)
        
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
    
    def subtrees(self, min_childs = 1):
        """
        Returns all subtrees in this node
        Arguments:
            min_childs: the minimal number of childs per tree
        """
        return [n for n in walk(self) if n.isLeaf() == False and len(n.children) >= min_childs]
    
    def dpaSubtrees(self):
        """
        Returns all leaves containing exactly one determiner (DT) in form of article [a, an, the]
        """
        known_indices = list()
        dpa_trees = list()
        subtrees = self.subtrees()
        for st in subtrees:
            if st.name == 'NP':
                dt_leaves = st.leavesWithPOS('DT')
                if len(dt_leaves) == 1 and all(dt_leaves[0].s_index != index for index in known_indices) and any(dt_leaves[0].name == name for name in ['a', 'an', 'the']):
                    dpa_trees.append(st)
                    known_indices.append(dt_leaves[0].s_index)
                
        return dpa_trees
                
    def deepNPSubtrees(self):
        """
        Returns list of deep NP subtrees which is the shortest ones inside complex NP
        """
        np_subtrees = list()
        subtrees = self.subtrees()
        for st in subtrees:
            if st.name == 'NP':
                if any(child.name == 'NP' for child in st.children):
                    # Intermediate NP
                    dpa_subtrees = st.dpaSubtrees() # N.B. can be optimized by direct check
                    if len(dpa_subtrees) == 0:
                        # Add NP as whole
                        np_subtrees.append(st)
                else:
                    # Deepest NP
                    np_subtrees.append(st)
                
        return np_subtrees
        
        
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
    print(node)
            
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
    tree_str = json.dumps(data[1]) # 723
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
    
    print("+++++++++++++++++++++ Deep NP Subtrees")
    subtrees = root.deepNPSubtrees()
    for st in subtrees:
        print("---")
        for l in st.leaves():
            printNode(l)

    print("+++++++++++++++++++++ Leaves with POS")
    leaves = root.leavesWithPOS('DT')
    for n in leaves:
        printNode(n)
            
    print("+++++++++++++++++++++ DPA Subtrees")
    subtrees = root.dpaSubtrees()
    for st in subtrees:
        print("---")
        for l in st.leaves():
            printNode(l)