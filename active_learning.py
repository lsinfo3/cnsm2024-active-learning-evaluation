# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:10:48 2023

@author: katha
"""
import al_strategies
import ml_helpers
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score
import copy
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
import warnings
import sys
import time
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
random.seed(42)

# Loads the (predefined) datasets and slightly preprocesses them (filling NaNs, INFs etc.)
def load_data(data_name):
    if data_name == "VPN": # 1st Application Detection Dataset
        file_path = 'TimeBasedFeatures-Dataset-15s-AllinOne.arff'
    elif data_name == "TOR": # 2nd Application Detection Dataset
        file_path = 'TimeBasedFeatures-15s-Layer2.arff'  
    elif data_name == "IOT": # Device Detection Dataset
        file_path = 'aggregated_data_devicetype.csv'
    elif data_name == "IDS": # Intrusion Detection Dataset
        file_path = 'Wednesday-workingHours.pcap_ISCX.csv'
    
    if data_name == "VPN" or data_name == "TOR":
        with open(file_path, 'r') as file:
            data = arff.loadarff(file)
        df_data = pd.DataFrame(data[0])
        df_data["class1"] = df_data["class1"].str.decode('utf-8') 
    elif data_name == "IOT":
        df_data = pd.read_csv(file_path)
        df_data = df_data.loc[:, ~df_data.columns.str.contains('^Unnamed')]
        df_data = df_data.drop(columns=["ip_dst_new", "source_port", "dest_port", "epoch_timestamp"], axis=1)
        #df_data = df_data.drop(columns=["epoch_timestamp"], axis=1)
    elif data_name == "IDS":
        df_data = pd.read_csv(file_path)
        df_data = df_data.loc[:, ~df_data.columns.str.contains('^Unnamed')]
        columns_to_drop = [" Timestamp", "Flow ID", " Source IP", " Source Port", " Destination IP", " Destination Port", " Fwd Header Length.1"] # last feature is apparently duplicate
        df_data = df_data.drop(columns=columns_to_drop, axis=1)
        #df_data.drop(df_data.columns[[0, 1, 2, 3, 4, 5, 6]], axis=1, inplace=True)

    df_data = df_data.fillna(0) # fill nans 
    df_data.replace(np.inf, 0 , inplace=True) # fill infs
    df_data = shuffle(df_data, random_state=42)
    df_data.reset_index(inplace=True, drop=True)
    
    return df_data



# The main method of this script:
# threshold depicts how strict the AL is
# strategy is one of the uncertainty measures (e.g., entropy)
# modus is pool or stream-based AL
# data_name is one of the four datasets
# admin_list contains a list of admin types, see below which ones we called for which dataset
# batch_size is how much data is used to make the batches from the training data
# sampling can be over, under or non to balance the dataset if wanted
# num_features is the number of features selected to train the model
# model can be any model from scikit-learn or similar, here it is an RF
def active_learning(threshold, strategy, modus, data_name, admin_list, num_batches=25, batch_size=0.99, num_folds=3, sampling = "none", num_features = 20, model = RandomForestClassifier(max_depth=20,verbose=0,n_jobs=3, random_state=42)):   
    # for reproducibility and if you just simulate one of the strategies since random decisions of virtual admins are decided by random numbers; 
    # otherwise results may ~slightly~ vary (shouldnt vary too much though, this is just for improved reproducibility)
    random.seed(69) 
    np.random.seed(69)
    results_all = []
    results=[]
    
    df_data = load_data(data_name)
    unique_labels = np.unique(df_data.iloc[:,-1:]) # Labels are usually in the last column
    kfold = KFold(n_splits = num_folds)
    for split, (train_index, test_index) in enumerate(kfold.split(df_data)):
    #for split in range(0, num_folds):
        print("#######################################"+strategy + " " + str(threshold) + " Fold:" + str(split))
        
        
        ########## optimal baseline start -> standard ML
        #train, test = train_test_split(df_data, test_size=0.2, random_state=split)
        model_optimum = copy.deepcopy(model)
        train, test = df_data.iloc[train_index], df_data.iloc[test_index]
        y_true, y_pred = ml_helpers.baseline_training(train, test, model_optimum, num_features = num_features, sampling = sampling)
        
        print("Optimum: " + str(f1_score(y_true, y_pred, average="micro")))
        results=results+[[split, threshold, strategy, modus, "False", 0,"optimum", accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="micro"),f1_score(y_true, y_pred, average="macro"), None]]
        ########## optimal baseline end     
        
        ######## initial baseline start -> only initial training data  
        model_baseline = copy.deepcopy(model)
        train, batches = train_test_split(train, test_size=batch_size, random_state=split)
        y_true, y_pred = ml_helpers.baseline_training(train, test, model_baseline, num_features = num_features, sampling = sampling)
        print("Baseline: " + str(f1_score(y_true, y_pred, average="micro")))  
        results=results+[[split, threshold, strategy, modus, "False", 0,"baseline", accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="micro"),f1_score(y_true, y_pred, average="macro"), None]]
        ######## initial baseline end
        
        ######## actual active learning starts here
        X_train_raw = train.iloc[:, :-1].values # initial training data
        y_test = test.iloc[:, -1] # test data labels, will not be touched during the AL other than measuring performance
        y_train = train.iloc[:, -1] # labels of initial training data
        X_test_raw = test.iloc[:, :-1].values # test data features, will not be added to the training at any point, only scaled etc.
        batches_split = np.array_split(batches, num_batches) # batches for incremental AL
        
        for admin in admin_list: # ["best", "better", "good", "mediocre", "bad", "worse", "worst", "adversarial", "none", "conf"]
            for self_train in [False, True]:
                # no self-training does not really make sense here
                if admin == "none" and self_train == False:
                    continue
                if admin == "conf" and self_train == False:
                    continue
                batch_nr = 1
                
                # copy to ensure the original data stays untouched
                X_retrain_raw = copy.deepcopy(X_train_raw)
                y_retrain = copy.deepcopy(y_train)
                model_retrain = copy.deepcopy(model)
                print("#######")
                print(admin)
                for batch in batches_split:
                    t = time.time() # log time to see how long it takes
                    y_batch = batch.iloc[:, -1] # labels
                    X_batch_raw = batch.iloc[:, :-1] # features
                    
                    # preprocess the data (scale, filter, select features etc.)
                    X_retrain, X_test, X_batch, y_retrain_resampled, y_test, y_batch = ml_helpers.batch_preprocessing(X_retrain_raw, X_test_raw, X_batch_raw, y_retrain, y_test, y_batch, num_features = num_features, sampling = sampling)
                    
                    # retrain the model with newly added data from the batches (in the first batch this is equal to the initial model, since the new data has not been added here)
                    model_retrain.fit(X_retrain, y_retrain_resampled)   
                    predicts_retrain = model_retrain.predict(X_test) # predict the labels of the test data as a measure of "progress"/effect of the AL throughout the batches
                    
                    probas = model_retrain.predict_proba(X_batch) # predict the label probabilities for the batches for measuring (in)confidence
                    predicts = model_retrain.predict(X_batch) # predict the labels of the batches to be (potentially) added to the retraining
                    
                    # conf is an array consisting of booleans that indicate if the decision for the samples were confident or not
                    conf, values = al_strategies.get_al_strategy(probas, threshold, strategy, modus, unique_labels)

                    relayed_inconf = None
                    y_retrain_list = []
                    X_retrain_raw_list = []

                    # just add all the labels the ML model put out to the retraining
                    if admin == "none":
                        y_retrain_list.append(predicts)
                        X_retrain_raw_list.append(X_batch_raw)
                    
                    # add only the confident labels to the retraining
                    elif admin == "conf":
                        y_retrain_list.append(predicts[conf])
                        X_retrain_raw_list.append(X_batch_raw[conf])
                    
                    # admin involvement
                    else:
                        if admin == "best":
                            wrong_decision_prob = 0
                        elif admin == "better":
                            wrong_decision_prob = 0.5/3
                        elif admin == "good":
                            wrong_decision_prob = 0.5/3*2
                        elif admin == "mediocre":
                            wrong_decision_prob = 0.5
                        elif admin == "bad":
                            wrong_decision_prob = 0.5+0.5/3
                        elif admin == "worse":
                            wrong_decision_prob = 0.5+0.5/3*2
                        elif admin == "worst":
                            wrong_decision_prob = 1.0
                
                
                        relayed_inconf = len(predicts[~conf]) # num. of relayed decisions to admin

                        conf_dec = predicts[conf]
                        inconf_dec = predicts[~conf]
                        
                        if self_train: # if self-training enabled: add them to the retraining data
                            y_retrain_list.append(predicts[conf])
                            X_retrain_raw_list.append(X_batch_raw[conf])
                            
                        
                        if admin != "adversarial": # adversarial admin is handled below
                            dec = np.random.uniform(0, 1, size=len(inconf_dec)) # generate random numbers between 0 and 1 for all inconf. decisions
                            correct_label = dec > wrong_decision_prob # if random number is higher than the admins right decision probability -> correct label
                            wrong_label = ~correct_label # inverse of the above
                            
                            # below: admin randomly chooses a label for the samples that he did not know the label for (as decided by the random number)
                            wrong_labels = np.random.choice(unique_labels,size=sum(wrong_label))
                            y_retrain_list.append(wrong_labels) # add retraining labels
                            X_retrain_raw_list.append(X_batch_raw[~conf][wrong_label]) # add retraining features; use "raw" data since we redo the scaling, selection etc.
                            
                            # now the same for correct decisions, but here he know the label, so no need to randomly choose a new one -> use ground truth
                            correct_labels = y_batch[~conf][correct_label]
                            y_retrain_list.append(correct_labels)
                            X_retrain_raw_list.append(X_batch_raw[~conf][correct_label])
                            
                        else: # adversarial admin purposefully chooses wrong label in a consistent manner (e.g., for class 1 he will always choose class 2, class 2 will be 3 etc.)
                            # https://stackoverflow.com/questions/48329879/find-index-for-multiple-elements-in-a-long-list
                            # here we basically just convert the labels into the indexes in the unique_labels array
                            unique_labels_dictionary = {item: idx for idx, item in enumerate(unique_labels)}
                            labels_idx = [unique_labels_dictionary.get(item) for item in y_batch[~conf]]
                            labels_idx = [(x+1)%len(unique_labels) for x in labels_idx] # increment the labels to simulate the above described adversarial behavior
                            
                            # now map back the indexes to labels (in dict. for performance)
                            unique_labels_dictionary2 = {idx: item for idx, item in enumerate(unique_labels)}
                            adversarial_labels = [unique_labels_dictionary2.get(item) for item in labels_idx]

                            # addd to retraining
                            y_retrain_list.append(adversarial_labels)
                            X_retrain_raw_list.append(X_batch_raw[~conf])
                                            
                    # don't forget to add the already existing retraining data
                    y_retrain_list.append(y_retrain)
                    X_retrain_raw_list.append(X_retrain_raw)   
                        
                    # convert the lists into a pandas dataframe
                    y_retrain = np.concatenate(y_retrain_list)  
                    X_retrain_raw = np.concatenate(X_retrain_raw_list)   
                    
                    # shuffle so no weird artifacts happen
                    y_retrain, X_retrain_raw = shuffle(y_retrain, X_retrain_raw, random_state=42)
    
                    # add to results
                    results.append([split, threshold, strategy, modus, self_train, batch_nr, admin, accuracy_score(y_test, predicts_retrain), f1_score(y_test, predicts_retrain, average="micro"),f1_score(y_test, predicts_retrain, average="macro"),relayed_inconf])
                    batch_nr = batch_nr + 1
                    print("Time needed in Batch: " + str(time.time()-t) + "s")
                    print("Batch:"+str(batch_nr-1) + " " + str(f1_score(y_test, predicts_retrain, average="micro")))
                
                #### final model; same as above in the loop -> since after the last batch the model is not evaluated anymore after adding the retraining data of last batch
                t = time.time()
                X_retrain, X_test, y_retrain_resampled, y_test = ml_helpers.simple_preprocessing(X_retrain_raw, X_test_raw, y_retrain, y_test, num_features = num_features, sampling = sampling)               
                model_retrain.fit(X_retrain, y_retrain_resampled)
                predicts_retrain = model_retrain.predict(X_test)
                results.append([split, threshold, strategy, modus, self_train, batch_nr, admin, accuracy_score(y_test, predicts_retrain), f1_score(y_test, predicts_retrain, average="micro"),f1_score(y_test, predicts_retrain, average="macro"),relayed_inconf])
                batch_nr = batch_nr + 1
                print("Time needed in Batch: " + str(time.time()-t) + "s")
                print("Batch:"+str(batch_nr-1) + " " + str(f1_score(y_test, predicts_retrain, average="micro")))
    results_df = pd.DataFrame(results,columns =['split','threshold','strategy', 'modus', 'selftrain','batch', 'admin', 'acc', 'f1micro', 'f1macro', 'relayeddec'])
    results_df.to_csv(os.path.join("results",data_name+"_"+str(threshold)+"_"+strategy+"_"+str(batch_size)+"_"+modus+"_"+str(num_batches)+".csv"), index=False)
    return results_all # this does not do anything anymore actually

# Below: for playing/testing around, uncomment this -> do this to understand the code
#active_learning(6/10, "DIFF", "stream", "TOR", admin_list=["best"], batch_size = 0.9, num_features = 20, sampling = "none")
#active_learning(8/10, "MANHAT", "stream", "VPN", admin_list=["best"], batch_size = 0.99, num_features = 20, sampling = "none")  
#active_learning(8/10, "DIFF", "stream", "IOT", admin_list=["best"], batch_size = 0.999, num_features = 10, sampling = "none") 
#active_learning(8/10, "MANHAT", "stream", "IDS", admin_list=["best"], batch_size = 0.9999, num_features = 10, sampling = "none")    

# Below: for command line calls and parallelism
if __name__ == "__main__":
    data_name = sys.argv[1]
    if data_name == "TOR":
        Parallel(n_jobs=15)(delayed(active_learning)((i+1)/10,j,k,"TOR", batch_size=0.9, admin_list=["best"], sampling = "none", num_features = 20) for (i,j,k) in tqdm(list(itertools.product(reversed(range(9)),  ["RND", "KS","WS","KL", "JS", "CHEBY", "EUCLID", "MANHAT", "MAX", "DIFF", "ENTROPY"], ["pool", "stream"]))) )
    elif data_name == "VPN":
        Parallel(n_jobs=7)(delayed(active_learning)((i+1)/10,j,k,"VPN", batch_size=0.99, admin_list=["best", "better", "good", "mediocre", "bad", "worse", "worst", "adversarial"], sampling = "none", num_features = 20) for (i,j,k) in tqdm(list(itertools.product(reversed(range(9)),  ["RND", "KS","WS","KL", "JS", "CHEBY", "EUCLID", "MANHAT", "MAX", "DIFF", "ENTROPY"], ["pool", "stream"]))) )
    elif data_name == "IOT":
        Parallel(n_jobs=15)(delayed(active_learning)((i+1)/10,j,k,"IOT", batch_size=0.999, admin_list=["best"], sampling = "none", num_features = 10) for (i,j,k) in tqdm(list(itertools.product(reversed(range(9)),  ["RND", "KS","WS","KL", "JS", "CHEBY", "EUCLID", "MANHAT", "MAX", "DIFF", "ENTROPY"], ["pool", "stream"]))) )
    elif data_name == "IDS":
        Parallel(n_jobs=7)(delayed(active_learning)((i+1)/10,j,k,"IDS", batch_size=0.9999, admin_list=["best"], sampling = "none", num_features = 10) for (i,j,k) in tqdm(list(itertools.product(reversed(range(9)),  ["RND", "KS","WS","KL", "JS", "CHEBY", "EUCLID", "MANHAT", "MAX", "DIFF", "ENTROPY"], ["pool", "stream"]))) )