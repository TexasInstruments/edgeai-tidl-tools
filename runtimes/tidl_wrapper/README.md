# TIDL Runtime Wrapper Libraries

This directory contains the wrapper runtime libraries for TIDL (Texas Instruments Deep Learning) acceleration. These libraries provide high-level interfaces for working with deep learning models on Texas Instruments devices.


## Directory Structure

- **cpp/**: C++ wrapper library implementations
  - **common/**: Shared utilities and data structures
  - **onnxrt/**: ONNX Runtime integration with TIDL
  - **tflitert/**: TensorFlow Lite Runtime integration with TIDL
  - **tidlrt/**: Direct TIDL Runtime interface

- **python/**: Python wrapper library implementations
  - **onnxrt/**: ONNX Runtime integration with TIDL
  - **tflitert/**: TensorFlow Lite Runtime integration with TIDL
  - **tvmrt/**: TVM Runtime integration with TIDL

## Getting Started

- [C++ Libraries](./cpp/README.md): C++ wrappers for runtime libraries, providing high-level interfaces for inference with deep learning models on Texas Instruments devices with TIDL acceleration.
- [Python Libraries](./python/README.md): Python wrappers for runtime libraries, providing high-level interfaces for model compilation and inference with TIDL acceleration.


For more information about wrapper libraries, please visit and go through the README of each.

For practical usage examples of these wrapper libraries, see the [examples](../examples/README.md) directory.
