# C++ Wrapper Libraries

This directory contains C++ wrappers for runtime libraries. These libraries provide high-level interfaces for inference with deep learning models on Texas Instruments devices with TIDL acceleration.

## Building the Libraries

The C++ wrapper libraries are built using CMake. The main CMakeLists.txt file in this directory configures the build process for all the libraries.

### Build Steps

```bash
rm -rf build bin lib
mkdir -p build
cd build
cmake ../
make

# These commands are invoked from runtimes/tidl_wrapper/cpp
```

After a successful build, the compiled library will be available as `lib/{build_type}/libtidl_wrapper.a`.

> [NOTE]
> For building on an x86 machine, make sure you have the TIDL_TOOLS_PATH environment variable set and edgeai-tidl-tools/tools/osrt_deps and edgeai-tidl-tools/tools/cnpy are present. If not, please run the setup scripts.

## Available Libraries

### Common

The [common](./common/README.md) directory contains shared utilities and data structures used across the C++ runtime implementations:

- **DlTensor**: A unified tensor structure for handling input and output tensor information

For detailed documentation on the common utilities, see the [Common README](./common/README.md).

### ONNXRT

The [ONNXRT](./onnxrt/README.md) module provides a C++ interface for working with ONNX models using TIDL module. It wraps the ONNX Runtime with TIDL provider to enable efficient model execution on TI devices.

**Key Features:**
- Load and run ONNX models with TIDL hardware acceleration
- Configure inference options for optimal performance
- Automatic tensor type conversion between ONNX and TIDL formats
- Detailed tensor information access
- Diagnostic output capabilities

For detailed documentation, examples, and API reference, see the [ONNXRT README](./onnxrt/README.md).

### TFLiteRT

The [TFLiteRT](./tflitert/README.md) module provides a C++ interface for working with TensorFlow Lite models using TIDL module. It integrates the TensorFlow Lite runtime with TIDL delegate for optimized execution on TI devices.

**Key Features:**
- Load and run TFLite models with TIDL hardware acceleration
- Configure inference options for optimal performance
- Automatic tensor type conversion between TFLite and TIDL formats
- Detailed tensor information access
- Diagnostic output capabilities

For detailed documentation, examples, and API reference, see the [TFLiteRT README](./tflitert/README.md).

### TIDLRT

The [TIDLRT](./tidlrt/README.md) module provides a C++ interface for working directly with TIDL module. It allows for direct access to TIDL interface without requiring an intermediate framework like ONNX Runtime or TensorFlow Lite.

**Key Features:**
- Direct access to TIDL hardware acceleration
- Compatible with artifacts generated from ONNXRT and TFLiteRT as long as all nodes are offloaded to TIDL
- Configure inference options for optimal performance
- Shared memory management for efficient data transfer
- Detailed tensor information access
- Diagnostic output capabilities

For detailed documentation, examples, and API reference, see the [TIDLRT README](./tidlrt/README.md).

### Examples

For practical usage examples, see the [C++ examples](../../examples/cpp/basic_example/README.md) directory.
