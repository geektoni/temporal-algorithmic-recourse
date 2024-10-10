#!/bin/bash

EXPERIMENT_DIR="./results/task_2_syn_data"

python test_causal_recourse.py --timesteps 100 --classifier dnn --scm linear --trend linear+seasonal --experiment only-robust --alpha 1.0 --output $EXPERIMENT_DIR
python test_causal_recourse.py --timesteps 100 --classifier dnn --scm non-linear --trend linear+seasonal --experiment only-robust --alpha 1.0 --output $EXPERIMENT_DIR

python test_causal_recourse.py --timesteps 100 --classifier dnn --scm linear --trend linear --experiment only-robust --alpha 1.0 --output $EXPERIMENT_DIR
python test_causal_recourse.py --timesteps 100 --classifier dnn --scm non-linear --trend linear --experiment only-robust --alpha 1.0 --output $EXPERIMENT_DIR

python test_causal_recourse.py --timesteps 100 --classifier dnn --scm linear --trend seasonal --experiment only-robust --alpha 1.0 --output $EXPERIMENT_DIR
python test_causal_recourse.py --timesteps 100 --classifier dnn --scm non-linear --trend seasonal --experiment only-robust --alpha 1.0 --output $EXPERIMENT_DIR

python test_causal_recourse.py --skip-ours --timesteps 100 --classifier dnn --scm linear --trend linear+seasonal --experiment only-robust-2 --alpha 1.0 --output $EXPERIMENT_DIR
python test_causal_recourse.py --skip-ours --timesteps 100 --classifier dnn --scm non-linear --trend linear+seasonal --experiment only-robust-2 --alpha 1.0 --output $EXPERIMENT_DIR

python test_causal_recourse.py --skip-ours --timesteps 100 --classifier dnn --scm linear --trend linear --experiment only-robust-2 --alpha 1.0 --output $EXPERIMENT_DIR
python test_causal_recourse.py --skip-ours --timesteps 100 --classifier dnn --scm non-linear --trend linear --experiment only-robust-2 --alpha 1.0 --output $EXPERIMENT_DIR

python test_causal_recourse.py --skip-ours --timesteps 100 --classifier dnn --scm linear --trend seasonal --experiment only-robust-2 --alpha 1.0 --output $EXPERIMENT_DIR
python test_causal_recourse.py --skip-ours --timesteps 100 --classifier dnn --scm non-linear --trend seasonal --experiment only-robust-2 --alpha 1.0 --output $EXPERIMENT_DIR
