#!/bin/bash

EXPERIMENT_DIR="./results/task_3_real_data"

python test_real_data.py --env adult --experiment only-robust --classifier dnn --learned --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
python test_real_data.py --env adult --skip-ours --experiment only-robust-2 --classifier dnn --learned --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
python test_real_data.py --env compas --experiment only-robust --classifier dnn --learned --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
python test_real_data.py --env compas --skip-ours --experiment only-robust-2 --classifier dnn --learned --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
python test_real_data.py --env loan --experiment only-robust --classifier dnn --learned --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
python test_real_data.py --env loan --skip-ours --experiment only-robust-2 --classifier dnn --learned --trend linear+seasonal --alpha 1.0 --output $EXPERIMENT_DIR
