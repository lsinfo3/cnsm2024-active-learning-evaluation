# -*- coding: utf-8 -*-
"""
Created on Thu May 16 07:13:17 2024

@author: katha
"""

import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from scipy import stats, special
import math

# from https://stackoverflow.com/questions/2739051/retrieve-the-two-highest-item-from-a-list-containing-100-000-integers
# we use this for the DIFF metric
def two_largest(inlist):
    """Return the two largest items in the sequence. The sequence must
    contain at least two items."""
    largest = 0 # NEW!
    second_largest = 0 # NEW!
    for item in inlist:
        if item > largest:
            largest = item
        elif largest > item > second_largest:
            second_largest = item
    # Return the results as a tuple
    return largest, second_largest

# used for most of the pool based strategies
def get_top_inconf_elements(confs, threshold_ranking):
    ind = np.argpartition(confs, threshold_ranking)[:threshold_ranking]
    conf = np.ones(confs.shape, dtype=bool) # https://stackoverflow.com/questions/64665409/fast-way-of-turning-a-list-of-indexes-into-a-boolean-mask
    conf[ind] = False # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    return conf

# just a wrapper for the KL measure
def kl_div_sum(x, y):
    return sum(special.kl_div(x, y))

# just a wrapper for the KS measure
def ks_stat(x, y):
    return stats.ks_2samp(x, y)[0]


# a helper function to get the distance to the uniform distribution, used in many of our measures
def get_dist_to_uniform_dist(dist_measure, probas, num_values):
    uniform_dist = [1/num_values]*num_values
    max_dist = dist_measure([1]+[0]*(len(uniform_dist)-1), uniform_dist)
    dist = np.array([dist_measure(x,uniform_dist) for x in probas])/max_dist
    return dist
    
# core of this script; contains all strategies we implemented to measure uncertainty
def get_al_strategy(probas, threshold, strategy, modus, unique_labels):
    # append some 0 if not all labels were initially contained; should not be the case for us since our training data is big enough
    if len(probas[0]) < len(unique_labels):
        probas=np.hstack((probas, np.zeros([len(probas), len(unique_labels)-len(probas[0])])))
    values = []
    uniform_dist = [1/len(unique_labels)]*len(unique_labels) # e.g., [0.25 0.25 0.25 0.25] for a problem with 4 classes
    # print(strategy)
    # print(modus)
    if modus == "stream": # stream-based AL
            if strategy == "MAX":
                # confidence via max. proba
                values = np.amax(probas, axis=1)  
                min_value = 1/len(unique_labels)
                values = (values - min_value)/(1 - min_value)
                conf = values > threshold
            elif strategy == "DIFF":
                # confidence via diff of two biggs probabilities
                values = np.abs(np.diff([two_largest(x) for x in probas])).flatten()
                conf = values > threshold
            elif strategy == "ENTROPY":
                # confidence via entropy   
                max_entrop = entropy([1]+[0]*(len(uniform_dist)-1), uniform_dist) # this is the same as just doing entropy(uniform_dist)=log(len(uniform_dist)), not sure why i made it more complicated...
                values = np.array([entropy(x) for x in probas])/max_entrop
                conf = values < threshold
            elif strategy in "KS":
                # confidence via ks dist
                values = get_dist_to_uniform_dist(ks_stat, probas, len(unique_labels))
                conf = values > threshold
            elif strategy == "WS":
                # confidence via ws dist
                values = get_dist_to_uniform_dist(wasserstein_distance, probas, len(unique_labels))
                conf = values > threshold
            elif strategy == "EUCLID":
                # confidence via euclidian dist
                values = get_dist_to_uniform_dist(distance.euclidean, probas, len(unique_labels))
                conf = values > threshold
            elif strategy == "MANHAT":
                # confidence via manhattan dist
                values = get_dist_to_uniform_dist(distance.cityblock, probas, len(unique_labels))
                conf = values > threshold
            elif strategy == "CHEBY":
                # confidence via chebyshev dist
                values = get_dist_to_uniform_dist(distance.chebyshev, probas, len(unique_labels))
                conf = values > threshold
            elif strategy == "KL":
                # confidence via kullbackleibler div     
                values = get_dist_to_uniform_dist(kl_div_sum, probas, len(unique_labels))
                conf = values > threshold
            elif strategy == "JS":
                # confidence via jensenshannon div
                values = get_dist_to_uniform_dist(distance.jensenshannon, probas, len(unique_labels))
                conf = values > threshold
            elif strategy == "RND":
                values = np.random.uniform(low=0, high=1, size=(len(probas),))
                # rnd_rows_index = np.random.choice([0, 1], size=len(probas), p=[threshold, 1-threshold]) # https://stackoverflow.com/questions/43065941/create-binary-random-matrix-with-probability-in-python
                # conf = np.array([x==1 for x in rnd_rows_index])
                conf = values > threshold
    elif modus=="pool": # pool-based AL
            threshold_ranking = math.ceil(len(probas)*threshold) # e.g., a threshold of 0.1 means that we relay the top 10% of inconfident samples to the admin
            if strategy == "MAX":
                # confidence via max. proba
                values = np.amax(probas, axis=1)
                min_value = 1/len(unique_labels)
                values = (values - min_value)/(1 - min_value)
                conf = get_top_inconf_elements(values, threshold_ranking)
            elif strategy == "DIFF":
                # confidence via diff of two biggs probabilities
                values = np.abs(np.diff([two_largest(x) for x in probas])).flatten()
                conf = get_top_inconf_elements(values, threshold_ranking)              
            elif strategy == "ENTROPY":
                # confidence via entropy
                max_entrop = entropy([1]+[0]*(len(uniform_dist)-1), uniform_dist) # this is the same as just doing entropy(uniform_dist)=log(len(uniform_dist)), not sure why i made it more complicated...
                values = np.array([entropy(x) for x in probas])/max_entrop
                ind = np.argpartition(values, -threshold_ranking)[-threshold_ranking:] # switch for entropy
                conf = np.ones(values.shape, dtype=bool) 
                conf[ind] = False         
            elif strategy == "KS":
                # confidence via ks dist
                values = get_dist_to_uniform_dist(ks_stat, probas, len(unique_labels))
                conf = get_top_inconf_elements(values, threshold_ranking)                
            elif strategy == "WS":
                # confidence via ws dist
                values = get_dist_to_uniform_dist(wasserstein_distance, probas, len(unique_labels))
                conf = get_top_inconf_elements(values, threshold_ranking)
            elif strategy == "EUCLID":
                # confidence via euclidian dist
                values = get_dist_to_uniform_dist(distance.euclidean, probas, len(unique_labels))
                conf = get_top_inconf_elements(values, threshold_ranking)
            elif strategy == "MANHAT":
                # confidence via manhattan dist
                values = get_dist_to_uniform_dist(distance.cityblock, probas, len(unique_labels))
                conf = get_top_inconf_elements(values, threshold_ranking)
            elif strategy == "CHEBY":
                # confidence via chebyshev dist
                values = get_dist_to_uniform_dist(distance.chebyshev, probas, len(unique_labels))
                conf = get_top_inconf_elements(values, threshold_ranking)
            elif strategy == "KL":
                # confidence via kullbackleibler div   
                values = get_dist_to_uniform_dist(kl_div_sum, probas, len(unique_labels))
                conf = get_top_inconf_elements(values, threshold_ranking)
            elif strategy == "JS":
                # confidence via jensenshannon div
                values = get_dist_to_uniform_dist(distance.jensenshannon, probas, len(unique_labels))
                conf = get_top_inconf_elements(values, threshold_ranking)
            elif strategy == "RND":
                values = np.random.uniform(low=0, high=1, size=(len(probas),))
                conf = get_top_inconf_elements(values, threshold_ranking)
                # rnd_rows_index = np.random.choice(len(probas),threshold_ranking,replace=False)
                # conf = np.ones(len(probas), dtype=bool) 
                # conf[rnd_rows_index] = False
    return (conf, values) # return chosen and unchosen (i.e., (in)confident) elements; 1 means conf: not relayed to admin, 0 means inconf: relayed