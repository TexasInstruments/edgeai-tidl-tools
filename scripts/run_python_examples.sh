#!/bin/bash
CURDIR=`pwd`
cd $CURDIR/examples/osrt_python/tfl
python3 tflrt_delegate.py -c
python3 tflrt_delegate.py
cd $CURDIR/examples/osrt_python/ort
python3 onnxrt_ep.py -c
python3 onnxrt_ep.py
cd $CURDIR/examples/osrt_python/tvm_dlr
python3  tvm-compilation-onnx-example.py --pc-inference
python3  tvm-compilation-tflite-example.py --pc-inference
python3  dlr-inference-example.py --pc-inference
cd $CURDIR




