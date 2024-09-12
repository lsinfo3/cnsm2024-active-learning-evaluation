# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:28:05 2024

@author: katha
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import VarianceThreshold

# Preprocessing for the stuff that happens during the AL part of the main code
# filters features that are always the same value
# scales the features to -1 to 1 range
# selects the best features
# (optional) resamples data for balance
# applies to batch data here too
def batch_preprocessing(X_train, X_test, X_batch, y_train, y_test, y_batch, num_features = 20, sampling = "none"):
       
        vfilter = VarianceThreshold()
        X_train = vfilter.fit_transform(X_train)
        X_test = vfilter.transform(X_test)
        
    
        scaler = MinMaxScaler()  
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        
        selector = SelectKBest(k=num_features)
        selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        

        
        if sampling == "over":
            ros = RandomOverSampler(random_state=0)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        elif sampling == "under":
            rus = RandomUnderSampler(random_state=0)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        
        X_batch = vfilter.transform(X_batch.values)
        X_batch = scaler.transform(X_batch)
        X_batch = selector.transform(X_batch)
        
        
        return X_train, X_test, X_batch, y_train, y_test, y_batch

# Preprocessing for the stuff that happens at the start and end of the code
# filters features that are always the same value
# scales the features to -1 to 1 range
# selects the best features
# (optional) resamples data for balance
# does not apply to batch data (either it is on of the baselines in the start, or the remainder evaluation in the end with no new batches after)
def simple_preprocessing(X_train, X_test, y_train, y_test, num_features = 20, sampling = "none"):
        vfilter = VarianceThreshold()
        X_train = vfilter.fit_transform(X_train)
        X_test = vfilter.transform(X_test)
    
        scaler = MinMaxScaler()  
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        
        selector = SelectKBest(k=num_features)
        selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)       
        
        if sampling == "over":
            ros = RandomOverSampler(random_state=0)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        elif sampling == "under":
            rus = RandomUnderSampler(random_state=0)
            X_train, y_train = rus.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test

# just some helper for the two baselines
# it preprocesses the data a bit and then calls the above function and returns the baseline predictions
def baseline_training(train, test, model, num_features = 20, sampling = "none"):
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    X_train, X_test, y_train, y_test = simple_preprocessing(X_train, X_test, y_train, y_test, num_features, sampling = sampling)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred 