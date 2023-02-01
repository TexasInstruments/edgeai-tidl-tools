#!/bin/bash

time=$(date)
echo "Current time is $time" > ~/report.txt 
cd /root/edgeai-tidl-tools
rm -r tidl_tools
rm -r tidl_tools.tar.gz
pwd
source ./setup.sh --skip_x86_python_install
mkdir build 
cd build
rm ../build/* -r
cmake ../examples
make -j
cd ..
source ./scripts/run_python_examples.sh
python3 ./scripts/gen_test_report.py
mkdir -p test_reports/$SOC/output_images
mv output_images/* test_reports/$SOC/output_images
mv test_report_pc.csv test_reports/$SOC/
mv ~/report.txt test_reports/$SOC/
# ls -l

