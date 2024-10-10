#!/bin/bash

EXPERIMENT_DIR="./results/task_0_alpha"

mkdir -p $EXPERIMENT_DIR

for alpha in "0.0" "0.3" "0.5" "0.7" "1.0"
do

    python test_causal_recourse.py --alpha ${alpha} --skip-ours --timesteps 50 --classifier dnn --scm linear --trend linear+seasonal --experiment only-robust --output $EXPERIMENT_DIR
    python test_causal_recourse.py --alpha ${alpha} --skip-ours --timesteps 50 --classifier dnn --scm non-linear --trend linear+seasonal --experiment only-robust --output $EXPERIMENT_DIR
    python test_real_data.py --env adult --alpha ${alpha} --skip-ours --timesteps 50 --classifier dnn --trend linear+seasonal --experiment only-robust --output $EXPERIMENT_DIR
    python test_real_data.py --env loan --alpha ${alpha} --skip-ours --timesteps 50 --classifier dnn --trend linear+seasonal --experiment only-robust --output $EXPERIMENT_DIR
    python test_real_data.py --env compas --alpha ${alpha} --skip-ours --timesteps 50 --classifier dnn --trend linear+seasonal --experiment only-robust --output $EXPERIMENT_DIR

done

for alpha in "0.0" "0.3" "0.5" "0.7" "1.0"
do

    python test_causal_recourse.py --alpha ${alpha} --skip-ours --timesteps 50 --classifier linear --scm linear --trend linear+seasonal --experiment only-robust --output $EXPERIMENT_DIR
    python test_causal_recourse.py --alpha ${alpha} --skip-ours --timesteps 50 --classifier linear --scm non-linear --trend linear+seasonal --experiment only-robust --output $EXPERIMENT_DIR

    python test_real_data.py --env adult --alpha ${alpha} --skip-ours --timesteps 50 --classifier linear --trend linear+seasonal --experiment only-robust --output $EXPERIMENT_DIR
    python test_real_data.py --env loan --alpha ${alpha} --skip-ours --timesteps 50 --classifier linear --trend linear+seasonal --experiment only-robust --output $EXPERIMENT_DIR
    python test_real_data.py --env compas --alpha ${alpha} --skip-ours --timesteps 50 --classifier linear --trend linear+seasonal --experiment only-robust --output $EXPERIMENT_DIR

done