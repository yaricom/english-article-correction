#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment with using extremely randomized tree classifier

@author: yaric
"""
import numpy as np
from scipy import sparse
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.externals import joblib
from sklearn import preprocessing

import data_set as ds
import config

# Parameters
n_classes = ds.DT.THE.value
n_estimators = 30
RANDOM_STATE = 123

def train():
    """
    Start model training
    Return:
        return trained model
    """
    corpora = ds.loadTrainCorpora()
    train_features = corpora["train"]["features"]
    train_labels = corpora["train"]["labels"]
    
    # NOTE: just probe for first 100 samples
    train_features = train_features[100:] 
    train_labels = train_labels[100:]
    
    # statndardize fetures
    train_features = preprocessing.scale(train_features)
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    print("Train features mean: %s, std: %s" % (mean, std))
    
    # train estimator
    clf = ExtraTreesClassifier(n_estimators = n_estimators, random_state = RANDOM_STATE, n_jobs = -1)
    model = clf.fit(train_features, train_labels)
    
    train_score = model.score(train_features, train_labels)
    
    # validate
    validate_features = corpora["validate"]["features"]
    validate_labels = corpora["validate"]["labels"]
    
    validate_features = preprocessing.scale(validate_features)
    
    train_valiate = model.score(validate_features, validate_labels)
    
    print("Train score: %.3f, validate score: %.3f" % (train_score, train_valiate))
    
    return model


def save_model(model, file):
    """
    Saves scikit-learn model to the file (filename.pkl)
    Arguments:
        model: the trained model
        file: the file to store model
    """
    joblib.dump(model, file)
    
def load_model(file):
    """
    Loads scikit-learn model from file (filename.pkl)
    Arguments:
        file: the file to read model from
    """
    return joblib.load(file)



if __name__ == '__main__':
    
    model = train()
    

