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
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import resample


def load_data():
    
    with open('features.pkl', 'rb') as file:
        X = pickle.load(file)
        X=np.array(X)

    with open('labels.pkl', 'rb') as file:
        Y = pickle.load(file)
        Y=np.array(Y)
    
    return X, Y


def up_df(table,col):
    
    df_majority = table[table[col]==0]
    df_minority = table[table[col]==1]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,   # sample with replacement
                                     n_samples=df_majority.shape[0]  # to match majority class
                                ) 
 
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    return df_upsampled
    
def upsampler(X,Y):

    columns_y=['y0','y1','y2','y3','y4','y5','y6','y7','y8','y9']
    y_df=pd.DataFrame(Y,columns=columns_y)
    
    columns_x=['age','gender','income','became_member_on']
    x_df=pd.DataFrame(X,columns=columns_x)
    
    table=pd.concat([x_df,y_df],axis=1)
    
    for col in columns_y:
        table=up_df(table,col)
    
    # Display new class counts
    #table['y0'].value_counts()
    scaler=MinMaxScaler()
    rescaled_data=scaler.fit_transform(table[columns_x])
    X_train, X_test, y_train, y_test = train_test_split(rescaled_data,\
                                                        table[columns_y], test_size=0.2)

    y_test=np.array(y_test)
    y_train=np.array(y_train)
    
    return X_train, X_test, y_train, y_test

def build_model():
    
    #pipeline model - parameters are not optimized in order to run faster
    pipeline = Pipeline([
        ('clf', MultiOutputClassifier(SVC(kernel='rbf',class_weight='balanced',\
                                          C=10.0, gamma='auto')))
    ])

    return pipeline

def optimize_model():
    
    #pipeline model
    pipeline = Pipeline([
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        #('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=100, learning_rate=0.01)))
        #('clf', MultiOutputClassifier(SVC(kernel='poly', C=100.0, gamma='auto')))
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
X_train, X_test, Y_train, Y_test = upsampler(X,Y)
        
print('Building model...')
model = build_model()
        
print('Training model...')
model.fit(X_train, Y_train)
        
print('Evaluating model...')
evaluate_model(model, X_test, Y_test)

print('Saving model...')
save_model(model)

print('Trained model saved!')
