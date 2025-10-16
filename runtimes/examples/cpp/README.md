# C++ Examples

This directory contains C++ examples demonstrating how to use the [C++ wrapper](../../tidl_wrapper/cpp/README.md) libraries for inference with deep learning models on Texas Instruments devices.

## Building the Examples

The C++ examples are built using CMake. The main CMakeLists.txt file in this directory configures the build process for all examples.

### Build Steps

Make sure you have [C++ wrapper](../../tidl_wrapper/cpp/README.md) libraries built before building the example. 

```bash
rm -rf build bin lib
mkdir -p build
cd build
cmake ../
make

# These commands are invoked from runtimes/examples/cpp
```

After a successful build, the compiled examples will be available in the `bin/{build_type}/{example}` directory.

> [NOTE]
> For building on an x86 machine, make sure you have the TIDL_TOOLS_PATH environment variable set and edgeai-tidl-tools/tools/osrt_deps and edgeai-tidl-tools/tools/cnpy are present. If not, please run the setup scripts.

## Available Examples

### Basic Example

The [Basic Example](./basic_example/README.md) demonstrates how to use ONNXRT, TFLiteRT, TVM RT and TIDLRT wrapper modules for model inference with TIDL acceleration. It provides a command-line interface to run models with various configuration options defined in config.yaml.

Before running the examples, set the TIDL_TOOLS_PATH and LD_LIBRARY_PATH environment variables:

Then, you can run the basic example:

```bash
./bin/Release/basic_example
```

For detailed documentation, usage instructions, and command-line options, see the [Basic Example README](./basic_example/README.md).

### Preemption Example

The [Preemption Example](./preemption_example/README.md) demonstrates priority scheduling mechanism of multiple networks on a single DSP core via pre-emption. Currently it only supports execution with TIDLRT.

Before running the examples, set the TIDL_TOOLS_PATH and LD_LIBRARY_PATH environment variables:

Then, you can run the preemption example:

```bash
./bin/Release/preemption_example
```

For detailed documentation, usage instructions, and command-line options, see the [Preemption Example README](./preemption_example/README.md).

## Utility Libraries

The [utils](./utils) directory contains utility functions and classes used by the examples:

- **argsparser.h**: Command-line argument parser
- **utils.h**: General utility functions
- **datasetloader**: Input data loading utilities for .bin and .npz files

These utilities provide common functionality that can be reused across different examples.
