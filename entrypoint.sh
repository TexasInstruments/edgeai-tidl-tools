#!/bin/bash

time=$(date)
echo "Current time is $time" > report.txt 
source ./setup.sh --skip_cpp_deps
source ./scripts/run_python_examples.sh
python3 ./scripts/gen_test_report.py
