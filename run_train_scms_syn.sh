#!/bin/bash

python train_scms_synthetic.py --env linear --classifier dnn --trend linear --retrain
python train_scms_synthetic.py --env linear --classifier dnn --trend seasonal --retrain
python train_scms_synthetic.py --env linear --classifier dnn --trend linear+seasonal --retrain

python train_scms_synthetic.py --env non-linear --classifier dnn --trend linear --retrain
python train_scms_synthetic.py --env non-linear --classifier dnn --trend seasonal --retrain
python train_scms_synthetic.py --env non-linear --classifier dnn --trend linear+seasonal --retrain

python train_scms_synthetic.py --env linear --classifier linear --trend linear --retrain
python train_scms_synthetic.py --env linear --classifier linear --trend seasonal --retrain
python train_scms_synthetic.py --env linear --classifier linear --trend linear+seasonal --retrain

python train_scms_synthetic.py --env non-linear --classifier linear --trend linear --retrain
python train_scms_synthetic.py --env non-linear --classifier linear --trend seasonal --retrain
python train_scms_synthetic.py --env non-linear --classifier linear --trend linear+seasonal --retrain