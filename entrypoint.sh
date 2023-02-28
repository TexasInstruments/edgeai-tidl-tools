#!/bin/bash

time=$(date)
echo "Current time is $time" > ~/report.txt 
cd /root/edgeai-tidl-tools
export ARM64_GCC_PATH=/root/edgeai-tidl-tools/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
rm -r tidl_tools
rm -r tidl_tools.tar.gz
rm output_images/* 
rm -r model-artifacts/*
pwd
if [ -z "$LOCAL_PATH" ];then
    source ./setup.sh --skip_x86_python_install
else
    source ./setup.sh --skip_x86_python_install --use_local
fi
mkdir build 
cd build
rm ../build/* -r
cmake ../examples
make -j
cd ..
source ./scripts/run_python_examples.sh
python3 ./scripts/gen_test_report.py
mkdir -p test_reports/$SOC/output_images
mkdir -p test_reports/$SOC/model-artifacts
rm test_reports/$SOC/output_images/*
rm test_reports/$SOC/test_report_pc_$SOC.csv
rm -r test_reports/$SOC/model-artifacts/*
cp output_images/* test_reports/$SOC/output_images/
cp test_report_pc_$SOC.csv test_reports/$SOC/
cp -r  model-artifacts/* test_reports/$SOC/model-artifacts/
mv ~/report.txt test_reports/$SOC/
# ls -l

