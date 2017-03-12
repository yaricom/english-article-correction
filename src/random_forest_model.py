#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment with using random forest classifier

@author: yaric
"""
import os
import shutil

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


import data_set as ds
import config

# Parameters
n_classes = ds.DT.THE.value
n_estimators = 100
RANDOM_STATE = 123

def train():
    """
    Start model training
    Return:
        return tuple with trained model and scaller used to scale input features
    """
    corpora = ds.loadTrainCorpora()
    train_features = corpora["train"]["features"]
    train_labels = corpora["train"]["labels"]
    
    # NOTE: just probe for first 100 samples
    #train_features = train_features[100:] 
    #train_labels = train_labels[100:]
    
    # statndardize fetures
    X_scaler = StandardScaler(with_mean = False) # we have sparse matrix - avoid centering
    X_train = X_scaler.fit_transform(train_features)
    
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    print("Train features mean: %s, std: %s" % (mean, std))
    
    # train estimator
    clf = RandomForestClassifier(n_estimators = n_estimators, random_state = RANDOM_STATE, n_jobs = -1)
    model = clf.fit(X_train, train_labels)
    
    train_score = model.score(train_features, train_labels)
    
    # validate
    validate_features = corpora["validate"]["features"]
    validate_labels = corpora["validate"]["labels"]
    
    X_test = X_scaler.transform(validate_features)
    valiate_score = model.score(X_test, validate_labels)
    
    print("Train score: %.3f, validate score: %.3f" % (train_score, valiate_score))
    
    return (model, X_scaler)

def predict(X, model, X_scaler):
    """
    Do prediction for provided fetures
    Arguments:
        X: the input features
        model: the classification predictive model
        X_scaler: the standard scaler used to scale train features
    Return:
        predicted labels as array of shape = [n_samples, n_classes] with probabilities
        of each class
    """
    X_test = X_scaler.transform(X)
    labels = model.predict_proba(X_test)
    return labels


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
    
    # Do train
    #
    model, x_scaler = train()
    
    # Create output directory
    if os.path.exists(config.models_dir) == True:
        shutil.rmtree(config.models_dir)  
    os.makedirs(config.models_dir)
    # save model and scaler 
    save_model(model, config.models_dir +  "/extra_trees_clf.pkl")
    save_model(x_scaler, config.models_dir + "/extra_trees_scaler.pkl")
    
    # Do prediction
    #
    test_features = np.load(config.test_features_path)
    labels = predict(test_features, model, x_scaler)
    
    print(labels)
    

