# Runtimes

This directory contains the wrapper libraries and examples for working with deep learning models on Texas Instruments devices.

## Directory Structure

- **tidl_wrapper/**: Wrapper over runtime libraries
  - **python/**: Python wrapper libraries for ONNX Runtime, TFLite Runtime and TVM Runtime
  - **cpp/**: C++ wrapper libraries for ONNX Runtime, TFLite Runtime, and direct TIDL Runtime

- **examples/**: Example applications demonstrating the use of wrapper APIs
  - **python/**: Python examples for model compilation and inference
  - **cpp/**: C++ examples for inference with TIDL acceleration

- **cmake/**: CMake build configuration file for C++ builds

## Getting Started

1. **Wrapper**: Start by exploring the [wrapper APIs over runtime libraries](./tidl_wrapper/README.md) to understand the available APIs for model compilation and inference.

2. **Examples**: Check out the [examples](./examples/README.md) to see practical demonstrations of how to use the TIDL runtime libraries.
