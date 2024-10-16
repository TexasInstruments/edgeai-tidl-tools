# Enabling backward compatibility

This section describes how to use an updated tools version (Say from SDK 10.x) with a lower SDK version (e.g. SDK 9.2). The tools version will clearly call out that if this is possible in the [Version Compatibility Table](../docs/version_compatibility_table.md). 

> **_NOTE:_**
> This is an experimental feature which has gone through limited validation
## Setup on Target Device
  - Follow the steps below on the target device to patch the **firmware**, **libraries**, **wheels**, **shared object files**.

  - **UPDATE_OSRT_COMPONENTS** and **UPDATE_FIRMWARE_AND_LIB** env variables can be set before running the script to control what to update.

 - UPDATE_OSRT_COMPONENTS replaces the following components on the target device filesystem and is set to **1** by default i.e it will replace the components:
    - onnxruntime, tflite_runtime and dlr python wheels
    - /usr/include/itidl_rt.h, /usr/include/itvm_rt.h, /usr/include/tensorflow, /usr/include/onnxruntime
    - /usr/lib/tflite_2.21, /usr/lib/libonnxruntime.so, /usr/lib/libtensorflow-lite.a

  - UPDATE_FIRMWARE_AND_LIB replaces the following components on the target device filesystem and is set to **0** by default i.e it will not replace the components unless explicitly set:
    - C7X firmwares under /lib/firmware
    - /usr/lib/libtivision_apps.so, /usr/lib/libvx_tidl_rt.so, /usr/lib/libtidl_onnxrt_EP.so, /usr/lib/libtidl_tfl_delegate.so

> **_NOTE:_**
>  Make sure you have stable internet connection on the EVM to run this script.


**Example usage for updating OSRT components (This assumes you are manually updating C7x firmwares with a patched SDK)**
```
export SOC=*soc*                    // [am62a,am68a,am68pa,am69a,am67a]
export TISDK_IMAGE=*adas or edgeai* // [adas for evm boards, edgeai for sk boards]
export UPDATE_OSRT_COMPONENTS=1
./setup_target_device.sh
```

**Example usage for updating OSRT components and C7x firmwares**
```
export SOC=*soc*                    // [am62a,am68a,am68pa,am69a,am67a]
export TISDK_IMAGE=*adas or edgeai* // [adas for evm boards, edgeai for sk boards]
export UPDATE_OSRT_COMPONENTS=1
export UPDATE_FIRMWARE_AND_LIB=1
./setup_target_device.sh
```


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