#!/bin/bash

python analytics/plot_sparsity.py results/task_2_syn_data/only-robust*_dnn_non-linear_*.pkl --synthetic
mv sparsity.pdf sparsity_non-linear.pdf

python analytics/plot_sparsity.py results/task_2_syn_data/only-robust*_dnn_linear_*.pkl --synthetic
mv sparsity.pdf sparsity_linear.pdf

python analytics/plot_sparsity.py results/task_3_real_data/only-robust*_actions_results.pkl
mv sparsity.pdf sparsity_real.pdf