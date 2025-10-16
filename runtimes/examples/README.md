# TIDL Runtime Examples

This directory contains examples demonstrating how to use the TIDL runtime libraries for model compilation and inference with deep learning models on Texas Instruments devices.

## Directory Structure

- **data/**: Downloaded as a part of setup. It contains sample models and input data used by the examples.
- **model-artifacts/**: Directory where compiled model artifacts are stored. This is a common directory shared between C++ and Python examples, allowing models compiled using Python to be used directly in C++ examples as well.
- **cpp/**: C++ example implementations.
- **python/**: Python example implementations.

## Getting Started

Each example directory contains detailed documentation on how to build and run the examples. For a quick start:

- [Python Examples](./python/README.md): Examples demonstrating model compilation and inference with Python APIs using ONNX Runtime, TFLite
Runtime, and TVM Runtime.
- [C++ Examples](./cpp/README.md): Examples demonstrating TIDL acceleration with C++ APIs using ONNX Runtime, TFLite Runtime, and direct TIDL Runtime.

Both Python and C++ examples demonstrate the basic workflow for using TIDL acceleration with deep learning models, including model loading, compilation (for Python), and inference with hardware acceleration. Models compiled using the Python examples can be directly used in the C++ examples without recompilation, as they share the same model-artifacts directory.

For more information about examples, please visit and go through the README of each.

For information on the wrapper APIs used by examples, see the [tidl_wrapper](../tidl_wrapper/README.md) directory.
