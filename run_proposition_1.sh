#!/bin/bash

python test_proposition_1.py --scm linear --classifier dnn
python test_proposition_1.py --scm "non-linear" --classifier dnn
python test_proposition_1.py --scm linear --classifier logistic
python test_proposition_1.py --scm "non-linear" --classifier logistic
