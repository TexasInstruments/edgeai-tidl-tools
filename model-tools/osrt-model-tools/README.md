# OSRT Model Tools

OSRT Model Tools is a collection of utilities for optimizing and modifying ONNX and TFLite models to improve their compatibility and performance with TIDL (TI Deep Learning).

## Overview

This package provides tools to help prepare models for efficient execution on TI devices. It includes utilities for both ONNX and TFLite model formats, allowing you to optimize models, convert between formats, and make specific modifications to improve performance.

## Setup
For setting up this package, execute the command from inside osrt-model-tools

```bash
    source ./setup.sh
```

This installs osrt-model-tools in your python environment as a pip package which can then be imported and used as usual.

## Components

### onnx_tools

The [onnx_tools](osrt_model_tools/onnx_tools/README.md) provide utilities for working with ONNX model format.

<!-- * Provides modules to perform various optimizations on ONNX models to make them more suitable for TIDL inference -->
* Provides various utility functions for ONNX models including model extractor, adding intermediate outputs, RGB to YUV input format convertor etc. 

### tflite_tools

The `tflite_tools` provide utilities for working with TensorFlow Lite model format.

* Provides modules to perform various optimizations on TensorFlow Lite models to make them more suitable for TIDL inference
* Provides various utility functions for TFLite models including RGB to YUV input format converter, optimizing model inputs etc.
