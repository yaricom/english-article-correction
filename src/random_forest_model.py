#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The predictor model based on RandomForestClassifier

@author: yaric
"""

from sklearn.ensemble import RandomForestClassifier

import config

RANDOM_STATE = 123

class RandomForest(object):
    
    def __init__(self, n_estimators = 300, model_path = config.models_dir +  "/random_forest/model.pkl", 
                 scaler_path = config.models_dir + "/random_forest/scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.n_estimators = n_estimators
    

    def train(self, X_train, labels):
        """
        Train model with given data corpus
        Arguments:
            X_train: the train data [n_samples, n_features]
            labels: the GT labels [n_samples, n_classes]
        Return:
            return trained model
        """
        # train estimator
        clf = RandomForestClassifier(n_estimators = self.n_estimators, random_state = RANDOM_STATE, n_jobs = -1)
        self.model = clf.fit(X_train, labels)
        
        train_score = self.model.score(X_train, labels)
        print("RandomForest:\ntrain score = %.3f, n_estimators = %d" % (train_score, self.n_estimators))
        
        return self.model



    

