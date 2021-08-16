#!/usr/bin/env bash

##################################################################
# setup the environment

# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

echo "Setting PSDK_BASE_PATH"
export PSDK_BASE_PATH="/mnt/ti-processor-sdk-rtos-j721e-evm-08_00_00_02"
echo "PSDK_BASE_PATH=${PSDK_BASE_PATH}"

echo "Setting TIDL_BASE_PATH"
export TIDL_BASE_PATH="${PSDK_BASE_PATH}/tidl_j7_08_00_00_03"
echo "TIDL_BASE_PATH=${TIDL_BASE_PATH}"
export TIDL_TOOLS_PATH=${TIDL_BASE_PATH}/"tidl_tools"
echo "TIDL_TOOLS_PATH=${TIDL_TOOLS_PATH}"

echo "Setting ARM64_GCC_PATH"
export ARM64_GCC_PATH="${PSDK_BASE_PATH}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
echo "ARM64_GCC_PATH=${ARM64_GCC_PATH}"

echo "Setting LD_LIBRARY_PATH"
import_path="${TIDL_BASE_PATH}/ti_dl/utils/tidlModelImport/out"
rt_path="${TIDL_BASE_PATH}/ti_dl/rt/out/PC/x86_64/LINUX/release"
tfl_delegate_path="${TIDL_BASE_PATH}/ti_dl/tfl_delegate/out/PC/x86_64/LINUX/release"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${import_path}:${rt_path}:${tfl_delegate_path}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

export TIDL_RT_PERFSTATS="1"
echo "TIDL_RT_PERFSTATS=${TIDL_RT_PERFSTATS}"

##################################################################
#jupyter notebook
