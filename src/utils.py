#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 21:40:08 2017

@author: yaric
"""

from enum import Enum

# The Part-of-Speech enumerations
class POS(Enum):
    CC = 1
    CD = 2
    DT = 3
    EX = 4
    FW = 5
    IN = 6
    JJ = 7
    JJR = 8
    JJS = 9
    LS = 10
    MD = 11
    NN = 12
    NNS = 13
    NNP = 14
    NNPS = 15
    PDT = 16
    POS = 17
    PRP = 18
    PRP_ = 19
    RB = 20
    RBR = 21
    RBS = 22
    RP = 23
    SYM = 24
    TO = 25
    UH = 26
    VB = 27
    VBD = 28
    VBG = 29
    VBN = 36
    VBP = 37
    VBZ = 38
    WDT = 39
    WP = 40
    WP_ = 41
    WRB = 42