# TIDLRT Wrapper

## Overview

The TIDLRT Wrapper module provides a C++ wrapper around the TIDL Runtime library, specifically designed for Texas Instruments devices. It simplifies the process of loading, initializing, and running inference on TIDL models with direct hardware acceleration on TI devices.

For more information underlying TIDL Runtime APIs, check [TIDLRT](../../../../docs/tidlrt.md)

## Compatibility with other runtime

Models compiled with any runtime (ONNX, TFLite, etc.) can be used with TIDLRT as long as the model is completely offloaded to the TIDL hardware accelerator. TIDLRT only requires the compiled network binary and IO descriptor binary files that are generated during the model compilation process.

### Identifying Completely Offloaded Models

To determine if a model is completely offloaded to TIDL hardware (and thus compatible with TIDLRT), you can check the model artifacts:

1. **Examine the compilation log**: During model compilation, check the log for messages indicating full offload

2. **Check the model artifacts folder**:
   - A completely offloaded model will have just a singular network and io binary file
   - tempDir/runtimes_visualization.svg will have all the nodes encapsulated under a single colored block

If your model meets these criteria, it can be used directly with TIDLRT for maximum performance and efficiency.

## Features

- Easy-to-use C++ interface for TIDL model inference
- Native TIDL hardware acceleration on TI devices
- Configurable inference options
- Detailed tensor information access
- Diagnostic output capabilities

## Class Structure

The module consists of two main classes:

1. **TIDLRT**: The main class that encapsulates TIDL Runtime functionality
2. **DlTensor**: A helper class that represents tensor data for inputs and outputs

## Dependencies

- TIDL Runtime library
- C++ Standard Library

## API Reference

### TIDLRT Class

#### Constructor

```cpp
TIDLRT()
```

Creates a new TIDLRT instance.

#### Methods

##### createInfer

```cpp
int32_t createInfer(std::map<std::string, std::string> &options)
```

Creates and initializes the inference engine with the specified options.

**Parameters:**
- `options`: Map of configuration options for the inference, including:

Refer to [Model Inference Options](../../../../docs/model_inference.md#inference-options) for details about available inference options.

**Returns:**
- Status code (0 for success, negative for failure)

> **Note**: If the artifacts folder contains multiple network binary files (files ending with _net.bin) or multiple IO descriptor binary files, createInfer will error out. Each artifacts folder should contain exactly one network binary file and one IO descriptor binary file.

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

##### dumpInfo

```cpp
void dumpInfo()
```

Prints detailed information about the model and its tensors.

### DlTensor Class

Find more information about DlTensor class [here](../common/README.md).

## Input/Output Memory Allocation and Padding

When working with tensors in tidlrt, it's important to understand how memory allocation and padding work:

1. **Buffer Size Allocation**: 
   - The total buffer size that needs to be allocated for each tensor comes from `tensor->allocSize` field
   - This size accounts for both the actual data and any required padding
   - The actual usable size for data operations is available in `tensor->validSize` field

2. **Input Data Handling**:
   - Input data must be populated respecting the padding requirements
   - The padding fields (`padT`, `padB`, `padL`, `padR`) indicate how many elements of padding are needed on each side
   - Failing to respect padding may result in incorrect inference results

3. **Output Data Handling**:
   - Output data will be produced respecting the same padding structure
   - When processing output data, you need to account for padding to extract the actual results
   - Use the shape and padding information to correctly interpret the output data

This padding is essential for optimal performance on TI hardware accelerators and ensures proper data alignment for efficient processing.

> **Important Note**: When using TIDLRT directly, you might notice that the total input buffer size allocates an extra (height × width) space. This additional memory is required for internal dataflow and optimization in the TIDL hardware accelerators. Always use the `tensor->allocSize` field for memory allocation rather than calculating the size manually to ensure proper operation. For data loading operations, use `tensor->validSize` which represents the actual usable size of the tensor.
> 
> **Example**: For a tensor with dimensions [1, 3, 224, 224] (batch=1, channels=3, height=224, width=224) and no padding:
> - Expected size calculation: 1 × 3 × 224 × 224 × elemSize = 150,528 bytes (for elemSize=1)
> - Actual allocation by TIDLRT (allocSize): 1 × 3 × 224 × 224 × elemSize + 224 × 224 × elemSize = 200,704 bytes
> - Usable size for data operations (validSize): 1 × 3 × 224 × 224 × elemSize = 150,528 bytes
> 
> This extra 50,176 bytes (224 × 224 × elemSize) is used by the TIDL hardware for optimized processing. The `tensor->allocSize` field will correctly account for this extra space, while `tensor->validSize` indicates how much of that space can be used for actual data.

### Visual Representation of Padding

To better understand how padding works in tensors, here are visual representations using symbols:

#### 2D Tensor with Padding (Height × Width)

```
┌───┬───┬───┬───┬───┐
│ P │ P │ P │ P │ P │  ← Row of top padding
├───┼───┼───┼───┼───┤
│ P │ D │ D │ D │ P │  ← First row of data with left/right padding
├───┼───┼───┼───┼───┤
│ P │ D │ D │ D │ P │  ← Second row of data with left/right padding
├───┼───┼───┼───┼───┤
│ P │ D │ D │ D │ P │  ← Third row of data with left/right padding
├───┼───┼───┼───┼───┤
│ P │ P │ P │ P │ P │  ← Row of bottom padding
└───┴───┴───┴───┴───┘

P = Padding element
D = Data element
```

In this representation:
- Top row: Top padding
- Bottom row: Bottom padding
- Left column: Left padding
- Right column: Right padding
- Center area: Actual data

#### 3D Tensor with Padding (Channel × Height × Width)

```
┌───┬───┬───┬───┬───┐
│ P │ P │ P │ P │ P │  ← Channel 0
├───┼───┼───┼───┼───┤
│ P │ D │ D │ D │ P │ 
├───┼───┼───┼───┼───┤
│ P │ D │ D │ D │ P │  
├───┼───┼───┼───┼───┤
│ P │ D │ D │ D │ P │  
├───┼───┼───┼───┼───┤
│ P │ P │ P │ P │ P │  
|---|---|---|---|---|
│ P │ P │ P │ P │ P │  ← Channel 1
├───┼───┼───┼───┼───┤
│ P │ D │ D │ D │ P │
├───┼───┼───┼───┼───┤
│ P │ D │ D │ D │ P │
├───┼───┼───┼───┼───┤
│ P │ D │ D │ D │ P │
├───┼───┼───┼───┼───┤
│ P │ P │ P │ P │ P │
└───┴───┴───┴───┴───┘

P = Padding element
D = Data element
```

#### Calculating Memory Offsets with Padding


To access elements correctly:
1. Skip top padding rows
2. For each data row, skip left padding before accessing data
3. After processing a row, skip right padding to move to the next row
4. After processing all data rows, skip bottom padding if moving to the next channel


```
For a tensor with dimensions [C, H, W] and padding:

rowStride = (W + padL + padR) * elemSize
channelStride = (H + padT + padB) * rowStride

To access element at position [c, h, w]:
1. Start at baseAddress
2. Add channel offset: c * channelStride
3. Skip top padding: padT * rowStride
4. Add row offset: h * rowStride
5. Skip left padding: padL * elemSize
6. Add column offset: w * elemSize

Final address = baseAddress + c * channelStride + (padT + h) * rowStride + (padL + w) * elemSize
```

> [NOTE] For detailed information on input/output tensor formats, refer to [docs/io_tensors.md](../../../../docs/io_tensors.md)

## Usage

For a complete working example, please refer to the basic example in [runtimes/examples/cpp/basic_example](../../../examples/cpp/basic_example/README.md)

> [NOTE] While running on x86, make sure to have TIDL_TOOLS_PATH set and add it to LD_LIBRARY_PATH environment variable as well.

### Simplified Example (No Padding)

If your model doesn't require padding or you're working with a simpler case, memory allocation and data handling become much more straightforward:

Note: The simplified approach above works when all padding values (`padT`, `padB`, `padL`, `padR`) are zero.

```cpp
#include "tidlrt_wrapper.h"
#include <vector>
#include <map>
#include <cstring>

int main()
{
    // Initialize the TIDLRT object
    tidlrt_wrapper::TIDLRT tidlrt;
    
    // Configure inference options
    std::map<std::string, std::string> options;
    options["artifacts_folder"] = "./artifacts";
    
    // Create the inference engine
    tidlrt.createInfer(options);
    
    // Get input and output tensor details
    const std::vector<DlTensor>* inputs = tidlrt.getInputDetails();
    const std::vector<DlTensor>* outputs = tidlrt.getOutputDetails();
    
    // Prepare input and output tensors
    std::vector<DlTensor*> inputTensors;
    std::vector<DlTensor*> outputTensors;
    
    // Allocate memory for input tensors
    for (size_t i = 0; i < inputs->size(); i++)
    {
        DlTensor* tensor = new DlTensor((*inputs)[i]);
        
        // Allocate memory using the allocSize field
        tensor->data = malloc(tensor->allocSize); // or use TIDLRT_allocSharedMem
        
        // When there's no padding, you can directly access the data as a contiguous block
        // For example, with a 3D tensor (channels, height, width):
        uint8_t* data = (uint8_t*)tensor->data;
        int channels = tensor->shape[3];
        int height = tensor->shape[4];
        int width = tensor->shape[5];
        
        // Simple case: No padding means contiguous memory layout
        // You can directly populate the data
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Calculate linear index in the tensor
                    int index = ((c * height + h) * width + w) * tensor->elemSize;
                    
                    // Populate your data
                    // For example: data[index] = your_input_data[c][h][w];
                }
            }
        }
        
        inputTensors.push_back(tensor);
    }
    
    // Allocate memory for output tensors
    for (size_t i = 0; i < outputs->size(); i++)
    {
        DlTensor* tensor = new DlTensor((*outputs)[i]);
        tensor->data = malloc(tensor->allocSize);
        outputTensors.push_back(tensor);
    }
    
    // Run inference
    tidlrt.runInfer(inputTensors, outputTensors);
    
    // Process output results
    for (size_t i = 0; i < outputTensors.size(); i++)
    {
        DlTensor* tensor = outputTensors[i];
        
        // When there's no padding, you can directly access the output data
        uint8_t* data = (uint8_t*)tensor->data;
        int channels = tensor->shape[3];
        int height = tensor->shape[4];
        int width = tensor->shape[5];
        
        // Process the output data
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Calculate linear index in the tensor
                    int index = ((c * height + h) * width + w) * tensor->elemSize;
                    
                    // Process your output data
                    // For example: your_output_data[c][h][w] = data[index];
                }
            }
        }
    }
    
    // Clean up
    for (auto tensor : inputTensors)
    {
        free(tensor->data); // or use TIDLRT_freeSharedMem
        delete tensor;
    }
    
    for (auto tensor : outputTensors)
    {
        free(tensor->data); // or use TIDLRT_freeSharedMem
        delete tensor;
    }
    
    return 0;
}
```

### Basic Usage Example (With Padding)

```cpp
#include "tidlrt_wrapper.h"
#include <vector>
#include <map>
#include <cstring>

int main()
{
    // Initialize the TIDLRT object
    tidlrt_wrapper::TIDLRT tidlrt;
    
    // Configure inference options
    std::map<std::string, std::string> options;
    options["artifacts_folder"] = "./artifacts";
    
    // Create the inference engine
    tidlrt.createInfer(options);
    
    // Get input and output tensor details
    const std::vector<DlTensor>* inputs = tidlrt.getInputDetails();
    const std::vector<DlTensor>* outputs = tidlrt.getOutputDetails();
    
    // Prepare input and output tensors
    std::vector<DlTensor*> inputTensors;
    std::vector<DlTensor*> outputTensors;
    
    // Allocate memory for input tensors
    for (size_t i = 0; i < inputs->size(); i++)
    {
        DlTensor* tensor = new DlTensor((*inputs)[i]);
        
        // Allocate memory using the allocSize field which accounts for padding
        tensor->data = malloc(tensor->allocSize); // or use TIDLRT_allocSharedMem
        
        // Populate input data respecting padding
        // For example, if we have a 3D tensor (channels, height, width) with padding:
        // Note: This is a simplified example. In practice, you would load actual data.
        uint8_t* baseData = (uint8_t*)tensor->data;
        int channels = tensor->shape[3];
        int height = tensor->shape[4];
        int width = tensor->shape[5];
        int padT = tensor->padT;
        int padB = tensor->padB;
        int padL = tensor->padL;
        int padR = tensor->padR;
        
        // Calculate row stride including padding
        int rowStride = (width + padL + padR) * tensor->elemSize;
        
        // Calculate channel stride including padding (height + top + bottom padding)
        int channelStride = (height + padT + padB) * rowStride;
        
        // For each channel
        for (int c = 0; c < channels; c++)
        {
            uint8_t* channelData = baseData + c * channelStride;
            
            // Skip top padding rows
            uint8_t* data = channelData + padT * rowStride;
            
            // For each actual row of data
            for (int h = 0; h < height; h++)
            {
                // Skip left padding
                uint8_t* rowData = data + padL * tensor->elemSize;
                
                // Fill actual data (simplified example)
                for (int w = 0; w < width; w++)
                {
                    // Populate your actual data here
                    // For example: 
                    // rowData[w * tensor->elemSize] = your_input_data;
                }
                
                // Move to next row (right padding is skipped implicitly)
                data += rowStride;
            }
        }
    
        
        inputTensors.push_back(tensor);
    }
    
    // Allocate memory for output tensors
    for (size_t i = 0; i < outputs->size(); i++)
    {
        DlTensor* tensor = new DlTensor((*outputs)[i]);
        
        // Allocate memory using the allocSize field which accounts for padding
        tensor->data = malloc(tensor->allocSize); // or use TIDLRT_allocSharedMem
        
        outputTensors.push_back(tensor);
    }
    
    // Run inference
    tidlrt.runInfer(inputTensors, outputTensors);
    
    // Process output results, accounting for padding
    for (size_t i = 0; i < outputTensors.size(); i++)
    {
        DlTensor* tensor = outputTensors[i];
        
        // Process output data respecting padding
        // Similar to input handling, you need to skip padding when accessing the data
        uint8_t* baseData = (uint8_t*)tensor->data;
        int channels = tensor->shape[3];
        int height = tensor->shape[4];
        int width = tensor->shape[5];
        int padT = tensor->padT;
        int padB = tensor->padB;
        int padL = tensor->padL;
        int padR = tensor->padR;
        
        // Calculate row stride including padding
        int rowStride = (width + padL + padR) * tensor->elemSize;
        
        // Calculate channel stride including padding (height + top + bottom padding)
        int channelStride = (height + padT + padB) * rowStride;
        
        // For each channel
        for (int c = 0; c < channels; c++)
        {
            uint8_t* channelData = baseData + c * channelStride;
            
            // Skip top padding rows
            uint8_t* data = channelData + padT * rowStride;
            
            // For each actual row of data
            for (int h = 0; h < height; h++)
            {
                // Skip left padding
                uint8_t* rowData = data + padL * tensor->elemSize;
                
                // Process actual data (simplified example)
                for (int w = 0; w < width; w++)
                {
                    // Process your output data here
                    // For example: 
                    // your_output_data = rowData[w * tensor->elemSize];
                }
                
                // Move to next row (right padding is skipped implicitly)
                data += rowStride;
            }
            
        }
    }
    
    // Clean up
    for (auto tensor : inputTensors)
    {
        free(tensor->data); // or use TIDLRT_freeSharedMem
        delete tensor;
    }
    
    for (auto tensor : outputTensors)
    {
        free(tensor->data); // or use TIDLRT_freeSharedMem
        delete tensor;
    }
    
    return 0;
}
```

## Inference Options

When using the `createInfer` method for model inference, you need to provide inference options to configure the TIDL inference process. Here are the key options:

- `artifacts_folder`: (Required) Path to the artifacts folder containing compiled model artifacts

Refer to [Model Inference Options](../../../../docs/model_inference.md#inference-options) for details about available inference options.

## Performance Metrics

The TIDLRT wrapper provides performance reporting capabilities through the `getPerformance()` method, which returns detailed metrics about model execution. This feature helps you analyze and optimize your models performance on TI hardware.

The method returns a map where:
- The key is the metric name
- The value is a pair containing:
  - The metric value (float)
  - The unit of measurement (string)

| Metric | Description | Unit |
|--------|-------------|------|
| `total_time` | Total time taken for model inference | milliseconds (ms) |
| `core_time` | Processing time excluding I/O copy operations | milliseconds (ms) |
| `graph_time` | Time spent in TIDL graph execution | milliseconds (ms) |
| `read_total` | Total DDR read bytes | bytes |
| `write_total` | Total DDR write bytes | bytes |