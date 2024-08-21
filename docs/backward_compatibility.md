# Enabling backward compatibility

## Setup on Target Device
  - Follow the steps below on the target device to patch the wheels, libraries and shared object files.
 - The script provided downloads and replaces the following components on the target device filesystem:
    - onnxruntime, tflite_runtime and dlr python wheels
    - /usr/include/tensorflow
    - /usr/include/itidl_rt.h
    - /usr/include/itvm_rt.h
    - /usr/lib/tflite_2.21
    - /usr/lib/libonnxruntime.so
    - /usr/lib/libtensorflow-lite.a

```
mv /usr/include/onnxruntime /usr/include/onnxruntime_old
mv /usr/include/tensorflow /usr/include/tensorflow_old
mv /usr/lib/tflite_2.21 /usr/lib/tflite_2.21_old
source ./setup_target_device.sh
```

> **_NOTE:_**
>  Make sure you have stable internet connection on the EVM to run this script.

## Compile and Validate on Target Device
- Once the setup is done, follow the steps below to build CPP application on EVM

```
mkdir build
cd build
cmake ../examples -DENABLE_SDK_9_2_COMPATIBILITY=1
make -j
cd ../
```
- Compile the models on X86_PC using the latest tidl-tools and copy over the artifacts to target device file system at ./edgeai-tidl-tools/
- Execute below to run inference on target device with both Python and CPP APIs

```
# scp -r <pc>/edgeai-tidl-tools/model-artifacts/  <dev board>/edgeai-tidl-tool/
# scp -r <pc>/edgeai-tidl-tools/models/  <dev board>/edgeai-tidl-tool/
python3 ./scripts/gen_test_report.py
```
- The execution of above step will generate output images at ```./edgeai-tidl-tools/output_images```.