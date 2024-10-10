#!/bin/bash

python train_scms_real.py --env adult --classifier linear --trend linear --retrain
python train_scms_real.py --env adult --classifier linear --trend seasonal --retrain
python train_scms_real.py --env adult --classifier linear --trend linear+seasonal --retrain

python train_scms_real.py --env loan --classifier linear --trend linear --retrain
python train_scms_real.py --env loan --classifier linear --trend seasonal --retrain
python train_scms_real.py --env loan --classifier linear --trend linear+seasonal --retrain

python train_scms_real.py --env compas --classifier linear --trend linear --retrain
python train_scms_real.py --env compas --classifier linear --trend seasonal --retrain
python train_scms_real.py --env compas --classifier linear --trend linear+seasonal --retrain