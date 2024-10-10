#!/bin/bash

EXPERIMENT_DIR="./results/task_3_real_data_ground_truth"

mkdir $EXPERIMENT_DIR

python test_real_data.py --start-t 50 --env adult --experiment only-robust --classifier dnn --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
python test_real_data.py --start-t 50 --env adult --skip-ours --experiment only-robust-2 --classifier dnn --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
python test_real_data.py --start-t 50 --env compas --experiment only-robust --classifier dnn --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
python test_real_data.py --start-t 50 --env compas --skip-ours --experiment only-robust-2 --classifier dnn --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
python test_real_data.py --start-t 50 --env loan --experiment only-robust --classifier dnn --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
python test_real_data.py --start-t 50 --env loan --skip-ours --experiment only-robust-2 --classifier dnn --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
