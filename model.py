# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:31:22 2020

@author: vinic_000
"""

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np


def load_data():
    
    with open('features.pkl', 'rb') as file:
        X = pickle.load(file)
        X=np.array(X)

    with open('labels.pkl', 'rb') as file:
        Y = pickle.load(file)
        Y=np.array(Y)
    
    return X, Y


def build_model():
    
    #pipeline model - parameters are not optimized in order to run faster
    pipeline = Pipeline([
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline

def optimize_model():
    
    #pipeline model
    pipeline = Pipeline([
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
         
    #grid serch parameters
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200], #set to 100
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test):
    #compute y_predicted
    y_pred = model.predict(X_test)
    #compute accuracy
    accuracy = (y_pred == Y_test).mean()
    print('Accuracy: {}'.format(accuracy))
    print('\n')
    #loop over the output columns to compute the f1 score, precision and recall
    columns=['y0','y1','y2','y3','y4','y5','y6','y7','y8','y9']
    y_pred=pd.DataFrame(y_pred,columns=columns)
    Y_test=pd.DataFrame(Y_test,columns=columns)
    for i in columns: 
        print('Column: {}'.format(i))
        print(classification_report(Y_test[i],y_pred[i]))
    return

def save_model(model):
    #open the file name and write the model
    with open('classifier.pkl', 'wb') as file:
        pickle.dump(model, file)
    return

X, Y = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
print('Building model...')
model = build_model()
        
print('Training model...')
model.fit(X_train, Y_train)
        
print('Evaluating model...')
evaluate_model(model, X_test, Y_test)

print('Saving model...')
save_model(model)

print('Trained model saved!')
