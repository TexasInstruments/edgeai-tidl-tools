#!/bin/bash

time=$(date)
echo "Current time is $time" > report.txt 
export DEVICE=j7
source ./setup.sh
mkdir build && cd build
cmake ../examples
make -j
cd ..
source ./scripts/run_python_examples.sh
python3 ./scripts/gen_test_report.py
ls -l

