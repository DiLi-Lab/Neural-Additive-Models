from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd


def evaluate_rf(X_train, X_val, X_test, y_train, y_val,
            param_grid = { 
                    'n_estimators': [500, 1000],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth' : [2,4,8,16,32, None],
                    'criterion' :['entropy'],
                    'n_jobs': [-1]
                }):
    """
    :input:     X_train: training data (matrix)
                X_val: validation data (matrix)
                X_test: test data (matrix)
                y_train: array with labels for train data
                y_val: array with labels for validation data
                param_grid: grid for grid search
                

    :return:    y_pred: predicted values for predicting on X_test
                best_parameters: dictionary containing the best hyper parameter combination (if param_grid is empty,
                best_parameters is an empty dict)
    """
    
    grid_search_verbosity = 0
    
    # combine train and test data, since we are using GridSearch
    X_train = np.vstack([X_train,X_val])
    y_train = np.concatenate([y_train,y_val])
    
    # rf
    rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, verbose = grid_search_verbosity)
    rf.fit(X_train, y_train)
    
    best_parameters = rf.best_params_
    pred_proba = rf.predict_proba(X_test)

    return pred_proba, best_parameters, rf