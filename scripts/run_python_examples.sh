#!/bin/bash
CURDIR=`pwd`
arch=$(uname -m)
if [[ $arch == x86_64 ]]; then
    echo "X64 Architecture"
elif [[ $arch == aarch64 ]]; then
    echo "ARM Architecture"
else
echo 'Processor Architecture must be x86_64 or aarch64'
echo 'Processor Architecture "'$arch'" is Not Supported '
return
fi

run_model=1
ncpus=1

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in    
    -o|--only_compile)
    run_model=0
    ;;
    -n=*|--ncpus=*)
    ncpus="${key#*=}"
    ;;
    -h|--help)
    echo Usage: $0 [options]
    echo
    echo Options,
    echo --only_compile            use this flag to generate model artifacts
    echo --ncpus=*                 use this to define number of threads. If not given, defaults to maximum cpu count.
    exit 0
    ;;
esac
shift # past argument
done
set -- "${POSITIONAL[@]}" # restore positional parameters
echo $run_model
echo --ncpus=$ncpus

if [ -z "$SOC" ];then
    echo "SOC not defined. Run either of below commands"
    echo "export SOC=am62"
    echo "export SOC=am62a"
    echo "export SOC=am68a"
    echo "export SOC=am68pa"
    echo "export SOC=am69a"
    echo "export SOC=am67a"
fi

if [[ $SOC == am68pa ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    if [[ $arch == x86_64 ]]; then
    python3 tflrt_delegate.py -c --ncpus=$ncpus
    fi
    if [ $run_model != 0 ];then
        echo "run python3 tflrt_delegate.py"
        python3 tflrt_delegate.py --ncpus=$ncpus
    fi    
    cd $CURDIR/examples/osrt_python/ort
    if [[ $arch == x86_64 ]]; then
    python3 onnxrt_ep.py -c --ncpus=$ncpus
    fi
    if [ $run_model != 0 ];then
        echo "run python3 onnxrt_ep.py"
        python3 onnxrt_ep.py --ncpus=$ncpus
    fi    
    # cd $CURDIR/examples/osrt_python/tvm
    # if [[ $arch == x86_64 ]]; then
    # python3 tvm_example.py -c --ncpus=$ncpus
    # fi
    # if [ $run_model != 0 ];then
    #     echo "run python3 tvm_example.py"
    #     python3 tvm_example.py --ncpus=$ncpus
    # fi  
    cd $CURDIR
elif [[ $SOC == am68a ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    if [[ $arch == x86_64 ]]; then
    python3 tflrt_delegate.py -c --ncpus=$ncpus
    fi
    if [ $run_model != 0 ];then
        echo "run python3 tflrt_delegate.py"
        python3 tflrt_delegate.py --ncpus=$ncpus
    fi
    cd $CURDIR/examples/osrt_python/ort
    if [[ $arch == x86_64 ]]; then
    python3 onnxrt_ep.py -c --ncpus=$ncpus
    fi
    if [ $run_model != 0 ];then
        echo "run python3 onnxrt_ep.py"
        python3 onnxrt_ep.py --ncpus=$ncpus
    fi        
    # cd $CURDIR/examples/osrt_python/tvm
    # if [[ $arch == x86_64 ]]; then
    # python3 tvm_example.py -c --ncpus=$ncpus
    # fi
    # if [ $run_model != 0 ];then
    #     echo "run python3 tvm_example.py"
    #     python3 tvm_example.py --ncpus=$ncpus
    # fi    
    cd $CURDIR
elif [[ $SOC == am69a ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    if [[ $arch == x86_64 ]]; then
    python3 tflrt_delegate.py -c --ncpus=$ncpus
    fi
    if [ $run_model != 0 ];then
        echo "run python3 tflrt_delegate.py"
        python3 tflrt_delegate.py --ncpus=$ncpus
    fi    
    cd $CURDIR/examples/osrt_python/ort
    if [[ $arch == x86_64 ]]; then
    python3 onnxrt_ep.py -c --ncpus=$ncpus
    fi
    if [ $run_model != 0 ];then
        echo "run python3 onnxrt_ep.py"
        python3 onnxrt_ep.py --ncpus=$ncpus
    fi 
    # cd $CURDIR/examples/osrt_python/tvm
    # if [[ $arch == x86_64 ]]; then
    # python3 tvm_example.py -c --ncpus=$ncpus
    # fi
    # if [ $run_model != 0 ];then
    #     echo "run python3 tvm_example.py"
    #     python3 tvm_example.py --ncpus=$ncpus
    # fi 
    cd $CURDIR    
elif [[ $SOC == am62 ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    if [ $run_model != 0 ];then
        echo "run python3 tflrt_delegate.py"
        python3 tflrt_delegate.py --ncpus=$ncpus
    fi 
    cd $CURDIR/examples/osrt_python/ort
    if [ $run_model != 0 ];then
        echo "run python3 onnxrt_ep.py"
        python3 onnxrt_ep.py --ncpus=$ncpus
    fi
    cd $CURDIR
elif [[ $SOC == am62a ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    if [[ $arch == x86_64 ]]; then
    python3 tflrt_delegate.py -c --ncpus=$ncpus
    fi
    if [ $run_model != 0 ];then
        echo "run python3 tflrt_delegate.py"
        python3 tflrt_delegate.py --ncpus=$ncpus
    fi 
    cd $CURDIR/examples/osrt_python/ort
    if [[ $arch == x86_64 ]]; then
    python3 onnxrt_ep.py -c --ncpus=$ncpus
    fi
    if [ $run_model != 0 ];then
        echo "run python3 onnxrt_ep.py"
        python3 onnxrt_ep.py --ncpus=$ncpus
    fi
    # cd $CURDIR/examples/osrt_python/tvm
    # if [[ $arch == x86_64 ]]; then
    # python3 tvm_example.py -c --ncpus=$ncpus
    # fi
    # if [ $run_model != 0 ];then
    #     echo "run python3 tvm_example.py"
    #     python3 tvm_example.py --ncpus=$ncpus
    # fi
    cd $CURDIR    
elif [[ $SOC == am67a ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    if [[ $arch == x86_64 ]]; then
    python3 tflrt_delegate.py -c --ncpus=$ncpus
    fi
    if [ $run_model != 0 ];then
        echo "run python3 tflrt_delegate.py"
        python3 tflrt_delegate.py --ncpus=$ncpus
    fi 
    cd $CURDIR/examples/osrt_python/ort
    if [[ $arch == x86_64 ]]; then
    python3 onnxrt_ep.py -c --ncpus=$ncpus
    fi
    if [ $run_model != 0 ];then
        echo "run python3 onnxrt_ep.py"
        python3 onnxrt_ep.py --ncpus=$ncpus
    fi
    cd $CURDIR 
fi





