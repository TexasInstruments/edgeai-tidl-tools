# Python Wrapper Libraries

This directory contains Python wrappers for runtime libraries. These libraries provide high-level interfaces for working with deep learning models on Texas Instruments devices with TIDL acceleration.

## Available Libraries

### ONNXRT

The [ONNXRT](./onnxrt/README.md) module provides a Python interface for working with ONNX models using TIDL acceleration. It wraps the ONNX Runtime with TIDL provider to enable efficient model execution on TI devices.

**Key Features:**
- Import and compile ONNX models for TIDL acceleration
- Run inference with TIDL hardware acceleration
- Collect detailed performance metrics
- Simple Python API for easy integration

For detailed documentation, examples, and API reference, see the [ONNXRT README](./onnxrt/README.md).

### TFLiteRT

The [TFLiteRT](./tflitert/README.md) module provides a Python interface for working with TensorFlow Lite models using TIDL acceleration. It integrates the TensorFlow Lite runtime with TIDL provider for optimized execution on TI devices.

**Key Features:**
- Import and compile TFLite models for TIDL acceleration
- Run inference with TIDL hardware acceleration
- Collect detailed performance metrics
- Simple Python API for easy integration

For detailed documentation, examples, and API reference, see the [TFLiteRT README](./tflitert/README.md).

### TIDLRT

The [TIDLRT](./tidlrt/README.md) module provides a Python interface for working with TIDL's native runtime. It enables direct access to TIDL's runtime capabilities without using open source runtimes as intermediaries.

**Key Features:**
- Import and compile models with TIDL's native runtime
- Run inference with TIDL's native runtime for maximum performance
- Collect detailed performance metrics
- Simple Python API for easy integration
- Support for padding information in tensor details
- Currently supports ONNX models only

For detailed documentation, examples, and API reference, see the [TIDLRT README](./tidlrt/README.md).

### TVMRT

The [TVMRT](./tvmrt/README.md) module provides a Python interface for working with TVM runtime using TIDL acceleration. This module enables easy model import and inference with TIDL offloading capabilities.

**Key Features:**
- Import and compile models with TVM runtime
- Run inference with TIDL hardware acceleration
- Collect detailed performance metrics
- Simple Python API for easy integration

For detailed documentation, examples, and API reference, see the [TVMRT README](./tvmrt/README.md).

### Examples

For practical usage examples, see the [Python examples](../../examples/python/basic_example/README.md) directory.
