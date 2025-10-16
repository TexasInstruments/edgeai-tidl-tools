# Dataset Loader

A flexible and extensible C++ module for loading input data from various sources.

## Overview

The Dataset Loader module provides a unified interface for loading input data from from different sources:

- Binary files
- NPZ files (NumPy's compressed archive format)

This module is particularly useful for machine learning workflows where you need to load input data for model inference.

## Components

### DatasetLoader Factory

The `DatasetLoader` class is a factory that creates the appropriate loader based on the specified type.

### Loader Types

#### BinLoader

Loads data from binary files. Supports continuous loading.

```cpp
#include "dataset_loader.h"

// Create baseloader with arguments
std::map<std::string, std::string> args = {{"file_path", "input_data.bin"}};
std::unique_ptr<DatasetLoaderBase> baseLoader = DatasetLoader::createLoader("bin", args);

// Downcast to BinLoader to access specific methods
BinLoader* binloader = dynamic_cast<BinLoader*>(baseLoader.get());
if (!binloader)
{
    std::cerr << "Failed to cast to BinLoader" << std::endl;
    return 1;
}

// Load data
float* buffer = new float[1 * 3 * 224 * 224];
size_t totalSize = (1 * 3 * 224 * 224) * sizeof(float);
binloader->load((void*)buffer, totalSize);

// Load next chunk
uint8_t* buffer2 = new uint8_t[1000];
size_t totalSize2 = (1000) * sizeof(uint8_t);
binloader->load((void*)buffer2, totalSize2);

// Reset the loader to start from the beginning
binloader->reset();

// Check how many bytes are remaining
size_t remaining_bytes = binloader->getRemainingBytes();

// Clean up
delete[] buffer;
delete[] buffer2;
```

Key features:
- Maintains a position pointer in the binary file therby allowing contiguous loading
- Provides option to reset and start loading from start of the file
- Memory-efficient as it only loads the requested portion of the binary file
- Provides option to load into pre-allocated memory for better memory management

#### NpzLoader

Loads data from NPZ files (NumPy's compressed archive format). Supports loading multiple arrays from a single file and handles padding for tensors.

```cpp
#include "dataset_loader.h"

// Create baseloader with arguments
std::map<std::string, std::string> args = {{"file_path", "input_data.npz"}};
std::unique_ptr<DatasetLoaderBase> baseLoader = DatasetLoader::createLoader("npz", args);

// Downcast to NpzLoader to access specific methods
NpzLoader* npzloader = dynamic_cast<NpzLoader*>(baseLoader.get());
if (!npzloader)
{
    std::cerr << "Failed to cast to NpzLoader" << std::endl;
    return 1;
}

// Load data without padding
float* buffer = new float[1 * 3 * 224 * 224];
size_t totalSize = (1 * 3 * 224 * 224) * sizeof(float);
npzloader->load((void*)buffer, totalSize);

// Load data with padding (e.g., 1 pixel on all sides)
// For a 3-channel 224x224 image, the padded size would be 3x226x226
float* paddedBuffer = new float[1 * 3 * 226 * 226];
size_t paddedSize = (1 * 3 * 226 * 226) * sizeof(float);
npzloader->load((void*)paddedBuffer, paddedSize, 1, 1, 1, 1); // padT=1, padB=1, padL=1, padR=1

// Load next array from the NPZ file
uint8_t* buffer2 = new uint8_t[1000];
size_t totalSize2 = (1000) * sizeof(uint8_t);
npzloader->load((void*)buffer2, totalSize2);

// Reset the loader to start from the beginning
npzloader->reset();

// Check how many items are remaining
size_t remaining_items = npzloader->getRemainingItems();

// Clean up
delete[] buffer;
delete[] paddedBuffer;
delete[] buffer2;
```

Key features:
- Loads data from NPZ files, which can contain multiple NumPy arrays
- **Important Note**: Data is loaded in sequence as arrays appear in the file, NOT based on the keys in the npz file. Make sure your arrays are in the correct order.
- Maintains a position pointer to cycle through arrays in the NPZ file
- Provides option to reset and start loading from the beginning of the file
- Validates data size to ensure compatibility with pre-allocated memory
- Automatically cycles through arrays when the end is reached
- **Supports padding for tensors** with configurable top, bottom, left, and right padding
- Handles multi-dimensional tensors:
  * Works with 1D tensors for left/right padding
  * Requires at least 2D tensors for top/bottom padding
  * Automatically adapts to 3D+ tensors (treating them as multi-channel 2D tensors)
- Zero-fills the padded regions automatically
- Provides detailed error messages for dimension mismatches

## Error Handling

All loaders include robust error handling:

- File not found errors for BIN and NPZ loaders
- Insufficient data checks for BIN loader
- Size mismatch checks for NPZ loader
- File format validation for NPZ loader
- Unsupported loader type errors for the factory class

## Extending the Dataset Loader

You can extend the dataset_loader module by creating your own custom loader. Here's a step-by-step guide:

### 1. Create a Loader Class

Create a new C++ header file (e.g., `custom_loader.h`) with a class that inherits from `DatasetLoaderBase`:

```cpp
#ifndef CUSTOM_LOADER_H
#define CUSTOM_LOADER_H

#include "dataset_loader_base.h"
#include <vector>
#include <string>

class CustomLoader : public DatasetLoaderBase
{
public:
    /**
     * @brief Construct a new CustomLoader object
     * 
     * @param filePath Path to the data file
     */
    CustomLoader(const std::string& filePath);
    
    /**
     * @brief Reset the loader state
     */
    void reset() override;
    
    /**
     * @brief Load data
     * 
     * @param data Pointer to pre-allocated memory where data will be stored
     * @param numBytes Number of bytes to load
     * @return size_t Number of bytes loaded
     */
    size_t load(void* data, size_t numBytes)
    
private:
    // Private implementation details
};

#endif // CUSTOM_LOADER_H
```

### 2. Integrate with the DatasetLoader Factory

Modify the `dataset_loader.h` file to include your custom loader:

```cpp
#include "custom_loader.h"

// In the DatasetLoader::createLoader method, add a case for your custom loader
if (loaderTypeLower == "custom")
{
    // Check for required arguments
    if (args.find("file_path") == args.end())
    {
        throw std::runtime_error("Missing required argument 'file_path' for CustomLoader");
    }
    
    return std::make_unique<CustomLoader>(args.at("file_path"));
}
```

### 3. Best Practices for Custom Loaders

1. **Consistent Interface**: Implement the same methods as the existing loaders (`load`, `reset`, etc.)
2. **Error Handling**: Include robust error checking for file existence, data shape, etc.
3. **Documentation**: Add clear comments explaining parameters and behavior
4. **Performance**: Consider memory and processing efficiency, especially for large datasets
5. **State Management**: Maintain proper state (e.g., current position) and provide methods to reset it
