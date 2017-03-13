#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The predictive models runner. Encapsulates common functionality which can be applied
to different predictors.

@author: yaric
"""
import os
import shutil
import argparse

import numpy as np
from sklearn.externals import joblib

from random_forest_model import RandomForest
import config

def predict(predictor_name, X_test, save_model = False, validate_model = True, save_labels = False):
    """
    Invoked to predict labels for provided test data features
    Arguments:
        predictor_name: the name of predictor to use
        X_test: the test data features [n_samples, n_features]
        save_model: flag to indicate whether to save trained model
        validate_model: flag to indicate whether to run trained model against validation data
        save_labels: the flag to indicate whether to save predicted labels array
    Return:
        tuple with predicted labels and validation score
    """
    corpora = __loadTrainCorpora()
    if predictor_name == 'RandomForest':
        predictor = RandomForest()
    else:
        raise Exception("Unknown predictor name: " + predictor_name)
    
    # train model
    model, X_scaler = predictor.train(corpora["train"]["features"], corpora["train"]["labels"])
    if save_model:
        __savePredictorModel(predictor)
    
    v_score = None    
    if validate_model:
        v_score = __validate(corpora["validate"]["features"], corpora["validate"]["labels"], model, X_scaler)
        print("validate score = %.3f" % (v_score))
    
    # predict
    labels = __predict(X_test, model, X_scaler)
    if save_labels:
        np.save(config.test_labels_prob_path, labels)
        print("Predicted labels saved to: " + config.test_labels_prob_path)
        
    return (labels, v_score)
    
def __validate(X, labels, model, X_scaler):
    """
    Run trained models against validation data
    Arguments:
        X: the test data [n_samples, n_features]
        labels: the GT labels [n_samples, n_classes]
        model: the prediction model
        X_scaler: the standard scaler to scale input features
    Return:
         the mean accuracy on the given test data and labels
    """
    X_test = X_scaler.transform(X)
    return model.score(X_test, labels)

def __predict(X, model, X_scaler):
    """
    Do prediction for provided fetures
    Arguments:
        X: the test data [n_samples, n_features]
        model: the classification predictive model
        X_scaler: the standard scaler used to scale train features
    Return:
        predicted labels as array of shape = [n_samples, n_classes] with probabilities
        of each class
    """
    X_test = X_scaler.transform(X)
    labels = model.predict_proba(X_test)
    return labels

def __savePredictorModel(predictor):
    """
    Saves trained model 
    """
    if predictor.model == None or predictor.X_scaler == None:
        raise Exception("Model not trained yet - nothing to save")
        
    # Create output directory
    model_dir = os.path.dirname(predictor.model_path)
    if os.path.exists(model_dir) == True:
        shutil.rmtree(model_dir)  
    os.makedirs(model_dir)
    
    # save model and scaler 
    joblib.dump(predictor.model, predictor.model_path)
    joblib.dump(predictor.X_scaler, predictor.scaler_path)
    
    print("Model saved to: " + model_dir)
    
def __loadPredictorModel(predictor):
    """
    Loads model from predefined location
    Return:
        loaded model
    """
    predictor.model = joblib.load(predictor.model_path)
    predictor.X_scaler = joblib.load(predictor.scaler_path)
    
def __loadTrainCorpora():
    # check that train corpora exists
    if any([os.path.exists(path) == False for path in 
            [config.train_features_path, config.train_labels_path, 
             config.validate_features_path, config.validate_labels_path]]):
        raise Exception("Necessary train corpora files not found")
        
    # Load data
    train_features = np.load(config.train_features_path)
    train_labels = np.load(config.train_labels_path)
    validate_features = np.load(config.validate_features_path)
    validate_labels = np.load(config.validate_labels_path)
    
    return {"train":{"features":train_features, "labels":train_labels},
            "validate":{"features":validate_features, "labels":validate_labels}}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='The predictive model runner')
    parser.add_argument('predictor_name', help='the name of predictor')
    parser.add_argument('--save_model', action='store_true', help='if set then trained model will be saved')
    parser.add_argument('--validate_model', action='store_true', help='if set then trained model will be validated against validate data')
    parser.add_argument('--save_labels', action='store_true', help='if set then predicted labels will be saved')
    args = parser.parse_args()
    
    # Do prediction
    #
    test_features = np.load(config.test_features_path)
    labels, _ = predict(args.predictor_name, test_features, 
                        save_model = args.save_model, 
                        validate_model = args.validate_model, 
                        save_labels = args.save_labels)
    
    
    
    
    