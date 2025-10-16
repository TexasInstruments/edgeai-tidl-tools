# Runtime Basic Example

This example demonstrates how to use ONNXRT, TFLiteRT, TVM RT and TIDLRT wrapper modules for model inference with TIDL acceleration. It provides a command-line interface to run ONNX and TFLite models with various configuration options.

## Overview

The `basic_example.cpp` showcases:

- Model inference with TIDL acceleration
- Support for both ONNX and TFLite models
- Model inference with TVM runtime.
- Handling different input data sources (.bin or .npz files)
- Processing multiple models defined in a configuration file
- Saving inference outputs as binary files
- Runtime-specific handling for different model formats
- Detailed model information display in verbose mode

## Requirements

- C++17 compatible compiler
- ONNX Runtime (for ONNX models)
- TFLite Runtime (for TFLite models)
- TVM Runtime (with TVM artifacts and ONNX model)
- YAML-CPP library
- CNPY library (for NPZ file loading)
- TIDL Tools (for model compilation)

## Directory Structure

### Example directory
```
runtimes/
├── examples/
│   ├── cpp/
│   │    ├── basic_example/           # This example
│   │    |   ├── basic_example.cpp    # Main example source
│   │    |   ├── basic_example.h      # Example header
│   │    |   ├── config.yaml          # Configuration file for models and options
│   │    |   ├── outputs/             # Output directory for inference results
│   │    |   └── README.md            # This file
|   |    |
|   |    └── utils/                   # Utility functions
│   |        ├── argsparser.h         # Command-line argument parser
│   |        ├── utils.h              # General utility functions
│   |        └── datasetloader/       # Input data loading utilities
│   |            ├── dataset_loader.h # Factory for creating data loaders
│   |            ├── bin_loader.h     # Binary file loader
│   |            └── npz_loader.h     # NumPy .npz file loader
│   |
│   ├── model-artifacts/              # Generated Model artifacts
│   │
│   └── data/
│       ├── inputs/                   # Sample input files
|       └── models/                   # Sample models
|
└── tidl_wrapper/
    └── cpp/
        ├── onnxrt/                   # ONNXRT wrapper module   
        ├── tflitert/                 # TFLiteRT wrapper module   
        └── tidlrt/                   # TIDLRT wrapper module
        └── tvmrt/                    # TVMRT wrapper module
```

## Command-Line Arguments

The program supports the following command-line arguments:

- `-h, --help`: Display help message and exit
- `-d, --disable_tidl_offload`: Disable offload to TIDL (runs on CPU only)
- `-v, --verbose`: Enable verbose output with detailed model and tensor information
- `-x, --config`: Path to config.yaml file (default: `<executable_dir>/config.yaml`)
- `-m, --models [MODEL_NAMES ...]`: Filter model keys to run from the config file
- `-r, --runtimes [RUNTIME_TYPES ...]`: Filter by runtime types ('onnxrt', 'tflitert', 'tvmrt' or 'tidlrt')

## Configuration File

The `config.yaml` file defines:

1. Global inference options applied to all models
2. List of models with their paths, runtime type, and input data sources
3. Model-specific inference options (which override global options)

**Important**: The name you give to each model in the configuration file is used to locate its artifacts directory. For each model, the program looks for artifacts in `<artifacts_base_dir>/<model_name>/artifacts/`, where `<model_name>` is the key used in the YAML configuration.

Example structure:

```yaml
infer_options:  # Global inference options
  "debug_level": 0
  # ... other options

models:
  # ONNX model example
  cl-ort-resnet18-v1:
    path: ../../data/models/resnet18_opset9.onnx
    inputs: ../../data/inputs/cl-ort-resnet18-v1.bin
    # Optional: Model-specific options that override global options
    infer_options:
      "option1": value1
  
  # TFLite model example
  cl-tfl-mobilenet_v1_1.0_224:
    path: ../../data/models/mobilenet_v1_1.0_224.tflite
    inputs: ../../data/inputs/cl-tfl-mobilenet_v1_1.0_224.npz
  
  # ONNX model directly with TIDLRT
  cl-ort-resnet18-v1:
    # Path is not required while running with tidlrt
    inputs: input_directory
    runtime: tidlrt

  # ONNX model with tvmrt based inference
  cl-tvm-ort-resnet18-v1:
    path: ../../data/models/resnet18_opset9.onnx
    inputs: ../../data/inputs/cl-ort-resnet18-v1.npz
    runtime: tvmrt
```

### Runtime Specification

`runtime` needs to be explicitly specified as `onnxrt`, `tflitert`, `tvmrt` or `tidlrt`

### Number of Frames

The `num_frames` option in the model configuration determines how many frames will be processed for each model. This value is determined as follows:

1. If `num_frames` is explicitly specified in the model configuration, that value is used
2. Otherwise, it defaults to the number of inputs provided in the `inputs` field

For example:
```yaml
models:
  model-1:
    path: ../../data/models/model1.onnx
    inputs: ../../data/inputs/input1.npz
    num_frames: 10  # Will process 10 frames, repeating the input if necessary
  
  model-2:
    path: ../../data/models/model2.tflite
    inputs: [input1.npz, input2.npz, input3.npz]  # Will process 3 frames by default

  model-3:
    path: ../../data/models/model3.onnx
    runtime: tvmrt
    inputs: [input1.npz, input2.npz, input3.npz]  # Will process 3 frames by default
```

## Usage Examples

### Environment Setup (Only for x86 runs)

Before running the example, set the TIDL_TOOLS_PATH and LD_LIBRARY_PATH environment variables:

```bash
export TIDL_TOOLS_PATH=/path/to/tidl_tools
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH
```

### Building the Example

```bash
cd runtimes
mkdir build && cd build
cmake ..
make
```

### Running the Example

To run all models defined in the config file:

```bash
./examples/cpp/basic_example/basic_example
```

To run specific models:

```bash
./examples/cpp/basic_example/basic_example --models cl-ort-resnet18-v1 cl-tfl-mobilenet_v1_1.0_224
```

To run models using ONNX runtime:

```bash
./examples/cpp/basic_example/basic_example --runtimes onnxrt
```

To run models using TFLite runtime:

```bash
./examples/cpp/basic_example/basic_example --runtimes tflitert
```

To run models using TIDL runtime:

```bash
./examples/cpp/basic_example/basic_example --runtimes tidlrt
```

To run models using TVM runtime:

```bash
./examples/cpp/basic_example/basic_example --runtimes tvmrt
```

To run with verbose output showing detailed model information:

```bash
./examples/cpp/basic_example/basic_example --verbose
```

To use a custom configuration file:

```bash
./examples/cpp/basic_example/basic_example --config /path/to/custom_config.yaml
```

### CPU-Only Inference

To run inference without TIDL acceleration (CPU only):

```bash
./examples/cpp/basic_example/basic_example --disable_tidl_offload
```

Note: TIDLRT models always run with TIDL acceleration and ignore the
`--disable_tidl_offload` flag.

TVMRT models expect this option to be specified as part of compilation
to generate relevant artifacts, this option does not take effect at
inference time.

## Workflow

1. **Initialization**:
   - Parses the config.yaml file
   - Determines runtime for each model (explicit or auto-detected)
   - Filters models based on runtime and model name if specified
   - Locates model artifacts based on the model name in the YAML configuration
   - The artifacts are expected to be in `<artifacts_base_dir>/<model_name>/artifacts/`

2. **Inference**:
   - Creates runtime-specific inference sessions
   - If verbose mode is enabled, displays detailed model and tensor information
   - Loads input data from .npz or .bin files
   - Runs inference on the models
   - Saves outputs as binary files

## Input Data

The example supports two types of input data:

1. **.npz files**: Loads input data from NumPy .npz files, supporting multiple arrays in a single file
2. **.bin files**: Loads input data from binary files, supporting continuous loading with different datatypes

Input data is specified in the config.yaml file for each model in several ways:

```yaml
models:
  model-name:
    # Single input file
    inputs: ../../data/inputs/input_data.npz  # Load from .npz file
    
    # Multiple input files as space-separated string
    inputs: ../../data/inputs/input1.npz ../../data/inputs/input2.npz
    
    # Multiple input files as a list
    inputs: [../../data/inputs/input1.bin, ../../data/inputs/input2.bin]
    
    # Directory containing input files
    inputs: ../../data/inputs/  # Will process all .bin and .npz files in this directory
```

The example supports both relative and absolute paths for input files. Relative paths are resolved relative to the location of the config.yaml file.

### .NPZ Files

When using .npz files as input data, the following requirements must be met:

1. **Multiple Inputs**: The NPZ loader supports cycling through multiple arrays in a single file. **Important Note: The data will be loaded in sequence as arrays appear in the file, NOT based on the keys in the npz file**. Make sure to have the same number of numpy arrays in the same order as the inputs in the model.

2. **Shape and Data Type**: Each array in the .npz file must have the same shape and data type as expected by the corresponding model input. For example, if the model expects a float32 tensor with shape (1, 3, 224, 224), the array in the .npz file must have the same shape and be of type np.float32.

3. **Padding Support**: The NPZ loader supports padding options for top, bottom, left, and right padding.

### .BIN Files

Binary files provide a more flexible and memory-efficient way to load input data, especially for large datasets. The BIN loader supports continuous loading from a long binary file with different datatypes for each call.

1. **Multiple Inputs**: The BIN loader supports continuous data loading from a single file. The loader maintains a position pointer in the binary file, allowing it to load data sequentially across multiple loads. For example, if the first call loads data of 100 bytes, the next call will load data starting from the 101st byte position.

2. **Shape and Data Type**: Each call can load the binary file into different data types and reshape it.

## Output Files

When running in inference mode, the program saves output tensors as binary files in:

- `<example_directory>/outputs/{model_name}/offload/frame_{frame_num}/` (when using TIDL acceleration)
- `<example_directory>/outputs/{model_name}/no_offload/frame_{frame_num}/` (when running on CPU only)

Where:
- `<example_directory>` is the directory containing the basic_example executable
- `{model_name}` is the name of the model as specified in the config file
- `frame_{frame_num}` is the frame number directory (e.g., "frame_1", "frame_2", etc.)

Output filenames follow the pattern:
```
{output_tensor_name}.bin
```

Where:
- `output_tensor_name` is the name of the output tensor from the model, with any '/' characters replaced by '_'. For example, if a model has an output tensor named `output/Softmax`, the output file would be named `output_Softmax.bin`.

All output tensors are automatically converted to float32 format before saving, regardless of their original data type. This ensures a consistent output format that can be easily processed by other tools or applications.

The program will display the full path of each saved output file at the end of inference execution.

## Verbose Mode

The `--verbose` or `-v` option enables detailed output about the model and its tensors. When this option is enabled, the program will call the `dumpInfo()` method of the runtime session after creating it, which displays:

- Model path
- Number of input tensors
- For each input tensor:
  - Name
  - Data type
  - Shape
  - Number of dimensions
  - Total number of elements
- Number of output tensors
- For each output tensor:
  - Name
  - Data type
  - Shape
  - Number of dimensions
  - Total number of elements

This information is particularly useful for:
- Debugging model loading issues
- Understanding the expected input and output formats
- Verifying tensor shapes and types

## Runtime-Specific Handling

The example handles runtime-specific differences transparently:

### ONNX Runtime:
- Uses ONNXRT class for model handling
- Uses ONNX-specific tensor details format
- Uses TIDLExecutionProvider for TIDL acceleration

### TFLite Runtime:
- Uses TFLiteRT class for model handling
- Uses TFLite-specific tensor details format
- Uses TFLite delegates for TIDL acceleration

### TIDL Runtime:
- Uses TIDLRT class for direct TIDL acceleration
- Provides a consistent interface with the other runtimes

### TVM Runtime
- Uses TVMRT class for model handling.
- Uses ONNX models as input and for tensor details format.
- The model-artifacts from TVM compilation is expected
  to specify various compilation options, including TIDL
  offload, c7x offload etc. The tvmrt wrapper uses these model
  artifacts to perform inference accordingly.

These differences are handled internally, providing a consistent user experience regardless of the model format being used.

## Memory Management

The example demonstrates proper memory management techniques:

1. **Aligned Memory Allocation**: Uses `posix_memalign` to ensure proper memory alignment for tensors
2. **Shared Memory**: Uses TIDL shared memory allocation when TIDL offload is enabled
3. **Proper Cleanup**: Ensures all allocated memory is properly freed after use

## Error Handling

The example includes robust error handling:

1. **Configuration Validation**: Validates all configuration parameters before running
2. **File Existence Checks**: Verifies that all model and input files exist
3. **Path Resolution**: Automatically resolves relative paths to absolute paths
4. **Size Validation**: Ensures input data size matches expected tensor sizes
5. **Exception Handling**: Uses try-catch blocks to handle exceptions gracefully
6. **Detailed Error Messages**: Provides specific error messages for different failure scenarios
