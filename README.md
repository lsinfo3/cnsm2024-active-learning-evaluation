# Certainly Uncertain: Demystifying ML  Uncertainty for Network Monitoring Tasks

## active_learning.py

Main script of the repository, which executes the AL loop with varying admin competence levels for all four datasets. You can download all datasets here: https://www.unb.ca/cic/datasets/

## al_strategies.py

Contains the different uncertainty measures to select the interesting data points, e.g., entropy or margin-based measures.

## ciciot_aggregator.py

The original preprocessed dataset of CICIOT comes with various .csv files. This script aggregates them all.

## ml_helpers.py

This file contains some helper functions for the main script.

## pareto_front_table.py

This script calculates the pareto front, w.r.t. to the sum of relayed decisions after simulation termination and F1-score. It calculates both the front and anti-front, and produces the table from the paper that counts how often a certain strategy is contained in a front.

## results

Contains the results after each iteration, including F1-scores, relayed decisions, self-training or not, etc; for the *VPN* dataset it contains the results for all admin competences, for the rest only for the expert admin.
