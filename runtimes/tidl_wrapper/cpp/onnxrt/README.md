# ONNXRT Wrapper

## Overview

The ONNXRT Wrapper module provides a C++ wrapper around the ONNX Runtime library, specifically designed for Texas Instruments devices. It simplifies the process of loading, initializing, and running inference on ONNX models with support for both CPU execution and TIDL (Texas Instruments Deep Learning) hardware acceleration.

## Features

- Easy-to-use C++ interface for ONNX model inference
- Support for TIDL hardware acceleration on TI devices
- Configurable inference options
- Automatic tensor type conversion between ONNX and TIDL formats
- Detailed tensor information access
- Diagnostic output capabilities

## Class Structure

The module consists of two main classes:

1. **ONNXRT**: The main class that encapsulates ONNX Runtime functionality
2. **DlTensor**: A helper class that represents tensor data for inputs and outputs

## Dependencies

- ONNX Runtime library
- TIDL provider for ONNX Runtime
- C++ Standard Library



## API Reference

### ONNXRT Class

#### Constructor

```cpp
ONNXRT(std::string modelPath, bool tidlOffload = true)
```

- `modelPath`: Path to the ONNX model file
- `tidlOffload`: Flag to enable TIDL hardware acceleration (default: true)

#### Methods

##### createInfer

```cpp
int32_t createInfer(std::map<std::string, std::string> &options)
```

Creates and initializes the inference engine with the specified options.

**Parameters:**
- `options`: Map of configuration options for the inference

Refer to [Model Inference Options](../../../../docs/model_inference.md#inference-options) for details about available inference options.

**Returns:**
- Status code (0 for success, negative for failure)

##### runInfer

```cpp
int32_t runInfer(const std::vector<DlTensor *> &inputs, std::vector<DlTensor *> &outputs)
```

Runs inference on the loaded model.

**Parameters:**
- `inputs`: Vector of input tensors containing the data for inference
- `outputs`: Vector of output tensors to store the inference results

**Returns:**
- Status code (0 for success, negative for failure)

##### getInputDetails

```cpp
const std::vector<DlTensor>* getInputDetails()
```

Gets details about the model's input tensors.

**Returns:**
- Pointer to vector of input tensor details

##### getOutputDetails

```cpp
const std::vector<DlTensor>* getOutputDetails()
```

Gets details about the model's output tensors.

**Returns:**
- Pointer to vector of output tensor details

```cpp
const std::map<std::string, std::pair<float, std::string>> ONNXRT::getPerformance()
```

Gets details about the inference time and memory usage.

**Returns:**
- Map containing performance metrics value and its unit

##### dumpInfo

```cpp
void dumpInfo()
```

Prints detailed information about the model and its tensors.

##### Static Utility Methods

```cpp
static int32_t Onnx2TidlType(const ONNXTensorElementDataType &onnxType, int32_t &tidlType)
static int32_t Tidl2OnnxType(const int32_t &tidlType, ONNXTensorElementDataType &onnxType)
```

Convert between ONNX and TIDL data types.

### DlTensor Class

Find more information about DlTensor class [here](../common/README.md).

## Input/Output Memory Allocation

When working with tensors in ONNXRT, follow these guidelines for memory allocation:

> **Important Note**: Always use the `tensor->allocSize` field for memory allocation rather than calculating the size manually. This ensures proper allocation of memory for the tensor data. For data loading operations, use `tensor->validSize` which represents the actual usable size of the tensor.
>
> Unlike TIDLRT, padding in ONNXRT is typically zero (`padT`, `padB`, `padL`, `padR` are usually 0), which means the memory layout is generally contiguous. This makes memory access more straightforward, but you should still use `tensor->size` for allocation.

## Usage

For a complete working example, please refer to the basic example in [runtimes/examples/cpp/basic_example](../../../examples/cpp/basic_example/README.md)

> [NOTE] While running on x86, make sure to have TIDL_TOOLS_PATH set and add it to LD_LIBRARY_PATH environment variable as well.

### Basic Usage Example

```cpp
#include "onnxrt_wrapper.h"
#include <vector>
#include <map>

int main() {
    // Initialize the ONNXRT object with model path and TIDL acceleration enabled
    onnxrt_wrapper::ONNXRT onnxrt("path/to/model.onnx", true);
    
    // Configure inference options
    std::map<std::string, std::string> options;
    options["artifacts_folder"] = "./artifacts";
    options["debug_level"] = "1";
    
    // Create the inference engine
    onnxrt.createInfer(options);
    
    // Get input and output tensor details
    const std::vector<DlTensor>* inputs = onnxrt.getInputDetails();
    const std::vector<DlTensor>* outputs = onnxrt.getOutputDetails();
    
    // Prepare input and output tensors
    std::vector<DlTensor*> inputTensors;
    std::vector<DlTensor*> outputTensors;
    
    // Allocate memory for input and output tensors
    // ... (allocation code)
    
    // Run inference
    onnxrt.runInfer(inputTensors, outputTensors);
    
    // Process results
    // ... (processing code)
    
    return 0;
}
```

## Inference Options

When using the `createInfer` method for model inference, you need to provide inference options to configure the TIDL inference process. Here are the key options:

- `artifacts_folder`: (Required) Path to the artifacts folder containing compiled model artifacts

Refer to [Model Inference Options](../../../../docs/model_inference.md#inference-options) for details about available inference options.

## Performance Metrics

The ONNXRT wrapper provides performance reporting capabilities through the `getPerformance()` method, which returns detailed metrics about model execution. This feature helps you analyze and optimize your models performance on TI hardware.

The method returns a map where:
- The key is the metric name
- The value is a pair containing:
  - The metric value (float)
  - The unit of measurement (string)

| Metric | Description | Unit |
|--------|-------------|------|
| `total_time` | Total time taken for model inference | milliseconds (ms) |
| `core_time` | Processing time excluding I/O copy operations | milliseconds (ms) |
| `subgraph_time` | Time spent in all TIDL subgraph execution | milliseconds (ms) |
| `read_total` | Total DDR read bytes | bytes |
| `write_total` | Total DDR write bytes | bytes |
