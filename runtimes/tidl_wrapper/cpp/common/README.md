# DlTensor - Unified Tensor Structure for EdgeAI TIDL Runtimes

## Overview

`dltensor.h` defines a unified tensor structure (`DlTensor`) that provides a common interface for handling input and output tensor information across all C++ runtime wrappers.

## Class Structure

The `DlTensor` class contains the following key components:

### Tensor Data

| Member | Type | Description |
|--------|------|-------------|
| `name` | `const char*` | Name of the tensor |
| `type` | `int32_t` | Unified type mapping to TIDL internal type |
| `typeName` | `std::string` | Human-readable string representation of the tensor's data type |
| `allocSize` | `int64_t` | Total size in bytes to be allocated, including padding |
| `validSize` | `int64_t` | Valid size in bytes that can be written, may be smaller than allocSize |
| `numElem` | `int64_t` | Total number of elements (product of all dimensions) |
| `elemSize` | `int32_t` | Size of each element in bytes |
| `numDim` | `int32_t` | Number of dimensions in the tensor |
| `shape` | `std::vector<int64_t>` | Vector containing the size of each dimension |
| `padT` | `int32_t` | Top padding of the tensor (>=2D tensor) |
| `padB` | `int32_t` | Bottom padding of the tensor (>=2D tensor) |
| `padL` | `int32_t` | Left padding of the tensor (>=1D tensor) |
| `padR` | `int32_t` | Right padding of the tensor (>=1D tensor) |
| `data` | `void*` | Pointer to the actual tensor data buffer |

### Utility Methods

| Method | Description |
|--------|-------------|
| `dumpInfo()` | Prints detailed information about the tensor to stdout |

## Usage

### Creating a DlTensor

```cpp
#include "common/dltensor.h"

// Create a new tensor
DlTensor tensor;
tensor.name = "input_tensor";
tensor.type = 1;  // Type depends on TIDL internal type mapping
tensor.typeName = "float32";  // Human-readable type name
tensor.numDim = 4;
tensor.shape = {1, 3, 224, 224};  // NCHW format example
tensor.elemSize = 4;  // 4 bytes for float32
tensor.numElem = 1 * 3 * 224 * 224;
tensor.allocSize = tensor.numElem * tensor.elemSize;
tensor.validSize = tensor.allocSize;  // For most cases, validSize equals allocSize
tensor.data = malloc(tensor.allocSize);
```

### Debugging Tensor Information

```cpp
// Print tensor information
tensor.dumpInfo();
```

Output:
```
Name          = input_tensor
Type          = float32
TIDL Type     = 1
Num Elements  = 150528
Element Size  = 4 bytes
Alloc Size    = 602112 bytes
Valid Size    = 602112 bytes
Num Dims      = 4
Shape         = [1, 3, 224, 224]
Pad Top       = 0
Pad Bottom    = 0
Pad Left      = 0
Pad Right     = 0
```

Note that padding values are important for memory layout, especially when using TIDLRT directly.
