#!/bin/bash

# Plot action analysis for real datasets
python analytics/plot_actions.py results/task_3_real_data/only-robust*_dnn_adult_linear+seasonal_100_1.0_10_250_20_actions_results.pkl
python analytics/plot_actions.py results/task_3_real_data/only-robust*_dnn_loan_linear+seasonal_100_1.0_10_250_20_actions_results.pkl
