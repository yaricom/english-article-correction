#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The predictor model based on RandomForestClassifier

@author: yaric
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import config

RANDOM_STATE = 123

class RandomForest(object):
    
    def __init__(self, n_estimators = 300, model_path = config.models_dir +  "/random_forest/model.pkl", 
                 scaler_path = config.models_dir + "/random_forest/scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.n_estimators = n_estimators
    

    def train(self, X, labels):
        """
        Train model with given data corpus
        Arguments:
            X: the train data [n_samples, n_features]
            labels: the GT labels [n_samples, n_classes]
        Return:
            return tuple with trained model and scaller used to scale input features
        """
        # statndardize fetures
        self.X_scaler = StandardScaler(with_mean = False) # we have sparse matrix - avoid centering
        X_train = self.X_scaler.fit_transform(X)
        
        #mean = X_train.mean(axis=0)
        #std = X_train.std(axis=0)
        #print("Train features mean: %s, std: %s\n" % (mean, std))
        
        # train estimator
        clf = RandomForestClassifier(n_estimators = self.n_estimators, random_state = RANDOM_STATE, n_jobs = -1)
        self.model = clf.fit(X_train, labels)
        
        train_score = self.model.score(X_train, labels)
        print("RandomForest:\ntrain score = %.3f, n_estimators = %d" % (train_score, self.n_estimators))
        
        return (self.model, self.X_scaler)



    

