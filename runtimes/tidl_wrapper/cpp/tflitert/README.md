# TFLiteRT Wraper

## Overview

The TFLiteRT Wrapper module provides a C++ wrapper around the TensorFlow Lite Runtime library, specifically designed for Texas Instruments devices. It simplifies the process of loading, initializing, and running inference on TensorFlow Lite models with support for both CPU execution and TIDL (Texas Instruments Deep Learning) hardware acceleration.

## Features

- Easy-to-use C++ interface for TensorFlow Lite model inference
- Support for TIDL hardware acceleration on TI devices
- Configurable inference options
- Custom memory allocation for optimized performance
- Automatic tensor type conversion between TensorFlow Lite and TIDL formats
- Detailed tensor information access
- Diagnostic output capabilities

## Class Structure

The module consists of two main classes:

1. **TFLITERT**: The main class that encapsulates TensorFlow Lite Runtime functionality
2. **DlTensor**: A helper class that represents tensor data for inputs and outputs

## Dependencies

- TensorFlow Lite Runtime library
- TIDL delegate for TensorFlow Lite
- C++ Standard Library
- Dynamic loading library (dlopen, dlsym)

## API Reference

### TFLITERT Class

#### Constructor

```cpp
TFLITERT(std::string modelPath, bool tidlOffload = true)
```

- `modelPath`: Path to the TensorFlow Lite model file (.tflite)
- `tidlOffload`: Flag to enable TIDL hardware acceleration (default: true)

#### Methods

##### createInfer

```cpp
int32_t createInfer(std::map<std::string, std::string> &options)
```

Creates and initializes the inference engine with the specified options.

**Parameters:**
- `options`: Map of configuration options for the inference. These options are passed to the TIDL delegate when enabled.

Refer to [Model Inference Options](../../../../docs/model_inference.md#inference-options) for details about available inference options.

**Returns:**
- Status code (0 for success, negative for failure)

**Throws:**
- `std::runtime_error` if model loading, interpreter creation, or tensor allocation fails

##### runInfer

```cpp
int32_t runInfer(const std::vector<DlTensor *> &inputs, std::vector<DlTensor *> &outputs)
```

Runs inference on the loaded model.

**Parameters:**
- `inputs`: Vector of input tensors containing the data for inference. The number and order must match the model's expected inputs.
- `outputs`: Vector of output tensors to store the inference results. The number and order must match the model's outputs.

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

##### dumpInfo

```cpp
void dumpInfo()
```

Prints detailed information about the model and its tensors, including model path, input/output tensor counts, and detailed information about each tensor.

##### Static Utility Methods

```cpp
static int32_t Tflite2TidlType(const TfLiteType &tfliteType, int32_t &tidlType)
static int32_t Tidl2TfliteType(const int32_t &tidlType, TfLiteType &tfliteType)
```

Convert between TensorFlow Lite and TIDL data types.


### DlTensor Class

Find more information about DlTensor class [here](../common/README.md).

## Input/Output Memory Allocation

When working with tensors in TFLITERT, follow these guidelines for memory allocation:

> **Important Note**: Always use the `tensor->allocSize` field for memory allocation rather than calculating the size manually. This ensures proper allocation of memory for the tensor data. For data loading operations, use `tensor->validSize` which represents the actual usable size of the tensor.
>
> Unlike TIDLRT, padding in TFLITERT is typically zero (`padT`, `padB`, `padL`, `padR` are usually 0), which means the memory layout is generally contiguous. This makes memory access more straightforward, and typically `allocSize` equals `validSize`. However, you should still use these fields appropriately for allocation and data loading operations.

## Usage

For a complete working example, please refer to the basic example in [runtimes/examples/cpp/basic_example](../../../examples/cpp/basic_example/README.md)

> [NOTE] While running on x86, make sure to have TIDL_TOOLS_PATH set and add it to LD_LIBRARY_PATH environment variable as well.

### Basic Usage Example

```cpp
#include "tflitert_wrapper.h"
#include <vector>
#include <map>

int main() {
    // Initialize the TFLITERT object with model path and TIDL acceleration enabled
    tflitert_wrapper::TFLITERT tflitert("path/to/model.tflite", true);
    
    // Configure inference options
    std::map<std::string, std::string> options;
    options["artifacts_folder"] = "./artifacts";
    options["debug_level"] = "1";
    
    // Create the inference engine
    tflitert.createInfer(options);
    
    // Get input and output tensor details
    const std::vector<DlTensor>* inputs = tflitert.getInputDetails();
    const std::vector<DlTensor>* outputs = tflitert.getOutputDetails();
    
    // Print model information (optional)
    tflitert.dumpInfo();
    
    // Prepare input and output tensors
    std::vector<DlTensor*> inputTensors;
    std::vector<DlTensor*> outputTensors;
    
    // Allocate memory for input and output tensors
    // ... (allocation code)
    
    // Fill input tensors with data
    // ... (data preparation code)
    
    // Run inference
    tflitert.runInfer(inputTensors, outputTensors);
    
    // Process results from output tensors
    // ... (processing code)
    
    return 0;
}
```
## Inference Options

When using the `createInfer` method for model inference, you need to provide inference options to configure the TIDL inference process. Here are the key options:

- `artifacts_folder`: (Required) Path to the artifacts folder containing compiled model artifacts

Refer to [Model Inference Options](../../../../docs/model_inference.md#inference-options) for details about available inference options.