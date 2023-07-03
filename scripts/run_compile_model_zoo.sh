#!/bin/bash
CURDIR=`pwd`
arch=$(uname -p)
if [[ $arch == x86_64 ]]; then
    echo "X64 Architecture"
elif [[ $arch == aarch64 ]]; then
    echo "ARM Architecture cannot used for compilation "
    exit 
else
    echo 'Processor Architecture must be x86_64 or aarch64'
    echo 'Processor Architecture "'$arch'" is Not Supported '
    exit
fi

if [ -z "$SOC" ];then
    echo "SOC not defined. Run either of below commands"
    echo "export SOC=am62"
    echo "export SOC=am62a"
    echo "export SOC=am68a"
    echo "export SOC=am68pa"
    echo "export SOC=am69a"
    exit
fi

echo "Assuming PSDK path as CWD $CURDIR"

echo $SOC
export TIDL_TOOLS_PATH=$CURDIR/c7x-mma-tidl/tidl_tools/
export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH
scripts_folder_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )


# compile zoo models
cd $scripts_folder_path/../examples/osrt_python/tfl
python3 tflrt_delegate.py  -z -c
cd $scripts_folder_path/../examples/osrt_python/ort
python3 onnxrt_ep.py -z -c





