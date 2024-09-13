# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:44:15 2023

@author: katha
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from paretoset import paretoset

admin_level = "best"

pareto_front_df_stream = pd.DataFrame()
pareto_front_df_pool = pd.DataFrame()
pareto_anti_front_df_stream = pd.DataFrame()
pareto_anti_front_df_pool = pd.DataFrame()

strategies=["RND", "MAX", "DIFF", "ENTROPY", "KS", "KL", "WS", "JS", "CHEBY", "EUCLID", "MANHAT"]

# ensure all strategies are contained in the subframes
pareto_front_df_stream = pareto_front_df_stream.reindex(strategies)
pareto_front_df_pool = pareto_front_df_pool.reindex(strategies)
pareto_anti_front_df_stream = pareto_anti_front_df_stream.reindex(strategies)
pareto_anti_front_df_pool = pareto_anti_front_df_pool.reindex(strategies)
        
for name in ["TOR", "VPN","IOT","IDS"]:
    if name == "TOR":
        test_size = 0.9
    if name == "VPN":
        test_size = 0.99
    if name == "IOT":
        test_size = 0.999
    if name == "IDS":
        test_size = 0.9999
    for eval_metric in ["f1micro"]:
        if eval_metric == "f1macro":
            yaxlabel = "F1-score (macro)"
        elif eval_metric == "f1micro":
            yaxlabel = "F1-score (micro)"
        for self_train in [False]:
            df_list =[]
            for strategy in strategies: 
                for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    for modus in ["stream", "pool"]:
                        #print(modus)
                        try:
                            results_df = pd.read_csv("results\\"+name+'_'+str(threshold)+'_'+strategy+'_'+str(test_size)+'_'+modus+'_25.csv')
                            df_list.append(results_df)
                        except:
                            continue
            
            results_all = np.concatenate(df_list) 
            results_df = pd.DataFrame(results_all, columns=['split','threshold','strategy', 'modus', 'selftrain','batch', 'admin', 'acc', 'f1micro', 'f1macro', 'relayeddec'])
           
            # relayed decisions column in the results after the last batch has the number of previous iteration.
            # should be 0, since there is no data left -> small correction here
            results_df[results_df.batch == 26] = results_df[results_df.batch == 26].assign(relayeddec=0)
            
            # filter out other admins and self_train modus
            results = results_df[results_df.admin != "baseline"]
            results = results[results.admin != "optimum"]
            results = results[results.selftrain == self_train]
            results = results[results.admin == admin_level]
            
            # aggregate over the three folds
            grouped_results = results.groupby(['modus', 'threshold', 'strategy', 'batch']).agg({eval_metric: ['mean'], 'relayeddec': ['mean']}).reset_index()
        

            # relayed decisions are per batch recorded, here we calculate the total sum
            cumulative_sum = 0
            previous_batch = None
            previous_threshold = None        
            results_agg = []
            
            for index, row in grouped_results.iterrows():
                current_batch = row['batch'][0]
                current_relayeddec = row['relayeddec'][0]
                current_threshold = row['threshold'][0]
                if (
                    previous_threshold is not None
                    and current_threshold == previous_threshold
                    and current_batch > previous_batch
                ):
                    cumulative_sum += current_relayeddec
                else:
                    cumulative_sum = current_relayeddec
            
                results_agg.append(cumulative_sum)           
                previous_batch = current_batch
                previous_threshold = current_threshold
            
            grouped_results['cumulativerelayeddec'] = results_agg # add new column with cumulative sum
            
            check_results=grouped_results # just for checking if the aggregation for cumulative relayed dec. worked
            
            grouped_results = grouped_results[grouped_results.batch == 26] # we only care about last batch then
            
        
            grouped_results[eval_metric]=grouped_results[eval_metric].round(4) # round slightly for pareto front... otherwise very slight increase/decrease of F1-score could be nonsensical, but doesnt really have an effect here.
            
            # filter out RANDOM sampling
            grouped_results_noRND=grouped_results[grouped_results.strategy!="RND"]
            
            # metrics for pareto frons are #decisions and f1-score
            strats =  (grouped_results_noRND[["cumulativerelayeddec", eval_metric]])
            
            # anti-front has inverted optimization goals
            anti_mask = paretoset(strats, sense=["max", "min"], distinct=False)
            
            # pareto front
            mask = paretoset(strats, sense=["min", "max"], distinct=False)
        
            # apply masks
            paretoset_strats = grouped_results_noRND[mask]
            anti_paretoset_strats = grouped_results_noRND[anti_mask]
            
            # get non pareto elements
            non_paretoset_strats = pd.concat([paretoset_strats,grouped_results, anti_paretoset_strats]).drop_duplicates(keep=False)
            
            
            # filter out 0 relayed dec. for table
            pareto_front_df_stream[name + "_stream"] = pd.Series(paretoset_strats[(paretoset_strats.modus == "stream") & (paretoset_strats.cumulativerelayeddec > 0)]["strategy"].value_counts())
            pareto_front_df_pool[name + "_pool"] = pd.Series(paretoset_strats[(paretoset_strats.modus == "pool") & (paretoset_strats.cumulativerelayeddec > 0)]["strategy"].value_counts())
            pareto_anti_front_df_stream[name + "_stream"] = pd.Series(anti_paretoset_strats[(anti_paretoset_strats.modus == "stream") & (anti_paretoset_strats.cumulativerelayeddec > 0)]["strategy"].value_counts())
            pareto_anti_front_df_pool[name + "_pool"] = pd.Series(anti_paretoset_strats[(anti_paretoset_strats.modus == "pool") & (anti_paretoset_strats.cumulativerelayeddec > 0)]["strategy"].value_counts())
                   
combined_df = pd.DataFrame() # merge all subframes

# put front and anti-front in one cell for stream-based
for column in pareto_front_df_stream.columns:
    dataset = column.split('_')[0]  # e.g., "IOT"
    combined_df[column] = pareto_front_df_stream[column].fillna('-').astype(str) + " " + pareto_anti_front_df_stream[column].fillna('-').astype(str)

# put front and anti-front in one cell for pool-based
for column in pareto_front_df_pool.columns:
    dataset = column.split('_')[0]  # e.g., "IOT"
    combined_df[column] = pareto_front_df_pool[column].fillna('-').astype(str) + " " + pareto_anti_front_df_pool[column].fillna('-').astype(str)

# calc. sum of fronts to put in last column
total_front_sum = pareto_front_df_stream.fillna(0).astype(float).sum(axis=1) + pareto_front_df_pool.fillna(0).astype(float).sum(axis=1)
total_anti_front_sum = pareto_anti_front_df_stream.fillna(0).astype(float).sum(axis=1) + pareto_anti_front_df_pool.fillna(0).astype(float).sum(axis=1)

# add final column to df
combined_df["Total (stream+pool)"] = (total_front_sum.astype(int).astype(str) + " " + total_anti_front_sum.astype(int).astype(str))


print(combined_df)
