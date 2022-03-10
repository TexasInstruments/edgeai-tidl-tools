#!/bin/bash

time=$(date)
echo "Current time is $time" > report.txt 
source ./setup.sh
mkdir build && cd build
cmake ../examples
cd ..
source ./scripts/run_python_examples.sh
python3 ./scripts/gen_test_report.py
ls -l

