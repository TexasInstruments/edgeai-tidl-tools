#!/bin/bash
CURDIR=`pwd`
arch=$(uname -p)
if [[ $arch == x86_64 ]]; then
    echo "X64 Architecture"
elif [[ $arch == aarch64 ]]; then
    echo "ARM Architecture"
else
echo 'Processor Architecture must be x86_64 or aarch64'
echo 'Processor Architecture "'$arch'" is Not Supported '
return
fi

if [ -z "$SOC" ];then
    echo "SOC not defined. Run either of below commands"
    echo "export SOC=am62"
    echo "export SOC=am62a"
    echo "export SOC=am68a"
    echo "export SOC=am68pa"
    echo "export SOC=am69a"
fi

if [[ $SOC == am68pa ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    if [[ $arch == x86_64 ]]; then
    python3 tflrt_delegate.py -c
    fi
    python3 tflrt_delegate.py
    cd $CURDIR/examples/osrt_python/ort
    if [[ $arch == x86_64 ]]; then
    python3 onnxrt_ep.py -c
    fi
    python3 onnxrt_ep.py
    cd $CURDIR/examples/osrt_python/tvm_dlr
    if [[ $arch == x86_64 ]]; then
    python3  tvm_compilation_onnx_example.py --pc-inference
    python3  tvm_compilation_tflite_example.py --pc-inference
    python3  tvm_compilation_onnx_example.py
    python3  tvm_compilation_tflite_example.py
    python3  tvm_compilation_timm_example.py
    fi
    python3  dlr_inference_example.py 
    cd $CURDIR
elif [[ $SOC == am68a ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    if [[ $arch == x86_64 ]]; then
    python3 tflrt_delegate.py -c
    fi
    python3 tflrt_delegate.py
    cd $CURDIR/examples/osrt_python/ort
    if [[ $arch == x86_64 ]]; then
    python3 onnxrt_ep.py -c
    fi
    python3 onnxrt_ep.py
    cd $CURDIR/examples/osrt_python/tvm_dlr
    if [[ $arch == x86_64 ]]; then
    python3  tvm_compilation_onnx_example.py --pc-inference
    python3  tvm_compilation_tflite_example.py --pc-inference
    python3  tvm_compilation_onnx_example.py
    python3  tvm_compilation_tflite_example.py
    python3  tvm_compilation_timm_example.py
    fi
    python3  dlr_inference_example.py 
    cd $CURDIR
elif [[ $SOC == am69a ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    if [[ $arch == x86_64 ]]; then
    python3 tflrt_delegate.py -c
    fi
    python3 tflrt_delegate.py
    cd $CURDIR/examples/osrt_python/ort
    if [[ $arch == x86_64 ]]; then
    python3 onnxrt_ep.py -c
    fi
    python3 onnxrt_ep.py
    cd $CURDIR/examples/osrt_python/tvm_dlr
    if [[ $arch == x86_64 ]]; then
    python3  tvm_compilation_onnx_example.py --pc-inference
    python3  tvm_compilation_tflite_example.py --pc-inference
    python3  tvm_compilation_onnx_example.py
    python3  tvm_compilation_tflite_example.py
    python3  tvm_compilation_timm_example.py
    fi
    python3  dlr_inference_example.py 
    cd $CURDIR    
elif [[ $SOC == am62 ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    python3 tflrt_delegate.py
    cd $CURDIR/examples/osrt_python/ort
    python3 onnxrt_ep.py 
    cd $CURDIR
elif [[ $SOC == am62a ]]; then
    cd $CURDIR/examples/osrt_python/tfl
    if [[ $arch == x86_64 ]]; then
    python3 tflrt_delegate.py -c
    fi
    python3 tflrt_delegate.py
    cd $CURDIR/examples/osrt_python/ort
    if [[ $arch == x86_64 ]]; then
    python3 onnxrt_ep.py -c
    fi
    python3 onnxrt_ep.py
    cd $CURDIR/examples/osrt_python/tvm_dlr
    if [[ $arch == x86_64 ]]; then
    python3  tvm_compilation_onnx_example.py --pc-inference
    python3  tvm_compilation_tflite_example.py --pc-inference
    python3  tvm_compilation_onnx_example.py
    python3  tvm_compilation_tflite_example.py
    python3  tvm_compilation_timm_example.py
    fi
    python3 dlr_inference_example.py
    cd $CURDIR    
fi





