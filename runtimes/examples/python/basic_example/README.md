# Runtime Basic Example

This example demonstrates how to use ONNXRT, TFLiteRT, TIDLRT, and TVMRT wrapper modules for model compilation and inference with TIDL acceleration. It provides a command-line interface to run ONNX, TFLite, TVM, and TIDL native runtime models with various configuration options.

## Overview

The `basic_example.py` script showcases:

- Model compilation with TIDL or TVM
- Model inference with TIDL acceleration
- Support for ONNX, TFLite models
- Processing multiple models defined in a configuration file
- Saving inference outputs as binary files
- Runtime-specific handling for different model formats
- Detailed model information display in verbose mode

## Requirements

- Python 3.10
- ONNX Runtime (for ONNX models)
- TFLite Runtime (for TFLite models)
- TIDL Runtime (for direct TIDL import)
- TVM Runtime (for TVM import)
- NumPy
- PyYAML
- TIDL Tools (for model compilation)

## Directory Structure

### Example directory
```
runtimes/
├── examples/
│   ├── python/
│   │    ├── basic_example/           # This example
│   │    |   ├── basic_example.py     # Main example script
│   │    |   ├── config.yaml          # Configuration file for models and options
│   │    |   ├── outputs/             # Output directory for inference results
│   │    |   └── README.md            # This file
|   |    |
|   |    └── utils/                   # Utility functions like dataset loader
│   |
│   ├── model-artifacts/              # Generated Model artifacts
│   │
│   └── data/
│       ├── inputs/                   # Sample input files
|       └── models/                   # Sample models
|
└── tidl_wrapper/
    └── python/
        ├── onnxrt/
        │   └── onnxrt_wrapper.py        # ONNXRT wrapper module
        ├── tflitert/
        │   └── tflitert_wrapper.py      # TFLiteRT wrapper module
        ├── tidlrt/
        │   └── tidlrt_wrapper.py        # TIDLRT wrapper module
        └── tvmrt/
            └── tvmrt_wrapper.py         # TVMRT wrapper module
```

## Command-Line Arguments

The script supports the following command-line arguments:

- `-c, --compile`: Run in model compilation mode
- `-i, --infer`: Run in inference mode (default if neither -c nor -i is specified)
- `-d, --disable_tidl_offload`: Disable offload to TIDL (runs on CPU only)
- `-v, --verbose`: Enable verbose output with detailed model and tensor information
- `-x, --config`: Path to config.yaml file (default: `<script_dir>/config.yaml`)
- `-m, --models [MODEL_NAMES ...]`: Filter model keys to run from the config file
- `-r, --runtimes [RUNTIME_TYPES ...]`: Filter by runtime types ('onnxrt', 'tflitert', 'tidlrt', or 'tvmrt')
- `--dump-frames N`: Collect and save only the first N frame outputs. Default: all frames

## Configuration File

The `config.yaml` file defines:

1. Global compilation/inference options applied to all models
2. List of models with their paths, runtime type, and input data sources
3. Model-specific compilation/inference options (which override global options)

Example structure:

```yaml
compile_options: # Global compilation options
  "tensor_bits": 8
  "accuracy_level": 1
  "advanced_options:calibration_frames": 1
  # ... other options

inference_options: # Global inference options
  "core_number": 1
  "advanced_options:temp_buffer_dir": /dev/shm
  # ... other options

models:
  # ONNX model example
  model-1:
    path: ../../../data/models/model.onnx
    runtime: onnxrt
    inputs: ../../../data/inputs/image.jpg
    num_frames: 5
    disable_onnx_optimizer: false  # Optional: Set to true to disable ONNX Runtime's internal optimization
  
  # TFLite model example
  model-2:
    path: ../../../data/models/model.tflite
    inputs: ../../../data/inputs/image.jpg
    # Model-specific options that override global options
    compile_options:
      "option1": value1
    inference_options:
      "option1": value1

  # ONNX model with TIDLRT example
  model-3:
    path: ../../../data/models/model.onnx
    runtime: tidlrt
    inputs: ../../../data/inputs/image.jpg
    
  # TFLite model with TIDLRT example
  model-4:
    path: ../../../data/models/model.tflite
    runtime: tidlrt
    inputs: ../../../data/inputs/image.jpg
```

### Runtime Specification

The `runtime` field in the model configuration determines which runtime will be used to process the model:

1. If `runtime` is explicitly specified as `onnxrt`, `tflitert`, `tidlrt`, or `tvmrt` that runtime will be used
2. If `runtime` is not specified, it will be automatically determined based on the file extension:
   - `.onnx` files will use the ONNXRT runtime
   - `.tflite` files will use the TFLiteRT runtime
   - For using TIDLRT or TVMRT, you must explicitly specify `runtime: tidlrt/tvmrt` as there is no file extension auto-detection

### Number of Frames

The `num_frames` option in the model configuration determines how many frames will be processed for each model. This value is determined as follows:

1. If `num_frames` is explicitly specified in the model configuration, that value is used
2. For compilation mode, if not specified, it defaults to the value of `advanced_options:calibration_frames`
3. Otherwise, it defaults to the number of inputs provided in the `inputs` field

For example:
```yaml
models:
  model-1:
    path: ../../../data/models/model1.onnx
    runtime: onnxrt
    inputs: ../../../data/inputs/image.jpg
    num_frames: 10  # Will process 10 frames, repeating the input if necessary

  model-2:
    path: ../../../data/models/model2.tflite
    runtime: tflitert
    inputs: [image.jpg, image2.jpg, image3.jpg]  # Will process 3 frames by default
```

If only one input is provided but multiple frames are requested, the same input will be used for all frames.

## Usage Examples

### Environment Setup (Only for x86 runs)

Before running the example, set the TIDL_TOOLS_PATH and LD_LIBRARY_PATH environment variable:

```bash
export TIDL_TOOLS_PATH=/path/to/tidl_tools
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH
```

### Model Compilation

To compile all models defined in the config file:

```bash
python basic_example.py --compile
```

To compile specific models:

```bash
python basic_example.py --compile --models cl-ort-resnet18-v1 od-tfl-ssd_mobilenet_v2_300_float
```

To compile with verbose output showing detailed model information:

```bash
python basic_example.py --compile --verbose
```

To compile for a specific platform i.e. 'pc' or 'evm' (valid only for TVMRT):

```bash
python basic_example.py --compile --compile_for_platform pc
```

### Model Inference

To run inference on all compiled models:

```bash
python basic_example.py --infer
```

To run inference on specific models:

```bash
python basic_example.py --infer --models cl-ort-resnet18-v1 cl-tfl-mobilenet_v1_1.0_224
```

To run inference with verbose output showing detailed model information:

```bash
python basic_example.py --infer --verbose
```

To use a custom configuration file:

```bash
python basic_example.py --config /path/to/custom_config.yaml
```

### CPU-Only Inference

To run inference without TIDL acceleration (CPU only):

```bash
python basic_example.py --infer --disable_tidl_offload
```

## Workflow

1. **Compilation Mode**:
   - Parses the config.yaml file
   - Determines runtime for each model (explicit or auto-detected)
   - Filters models based on runtime and model name if specified
   - Creates and cleans up artifacts folders
   - Runs shape inference on ONNX models
   - Creates runtime-specific import sessions with compilation options
   - If verbose mode is enabled, displays detailed model and tensor information
   - Runs import for each model with calibration data
   - Generates model artifacts in the artifacts folder

2. **Inference Mode**:
   - Parses the config.yaml file
   - Determines runtime for each model (explicit or auto-detected)
   - Filters models based on runtime and model name if specified
   - Checks if model artifacts exist
   - Creates runtime-specific inference sessions
   - If verbose mode is enabled, displays detailed model and tensor information
   - Runs inference on the models
   - Collects performance metrics
   - Saves outputs as binary files

## Input Data

The example supports four types of input data:

1. **Image**: Loads input data from Image (.jpg, .jpeg, .png). Supports both single images and batches of images.
2. **Random**: Generates random input data (seed = 0) with the appropriate shape and data type
3. **.npz files**: Loads input data from NumPy .npz files by matching array keys to model input names, with sequential index-based fallback
4. **.bin files**: Loads input data from binary files. Binary files contain raw tensor data that can be directly loaded into model inputs.

Input data is specified in the config.yaml file for each model in several ways:

```yaml
models:
  model-name:
    # Random data
    inputs: "random"  # Use random data (seed = 0)

    # Single input file
    inputs: ../../../data/inputs/image.jpg 
    
    # Multiple input files as a list
    inputs: ["../../../data/inputs/image.jpg", "../../../data/inputs/image2.jpg"]
    
    # Control ONNX Runtime optimizations (for ONNX models only)
    disable_onnx_optimizer: true  # Disables ONNX Runtime internal optimizations
```

The example supports both relative and absolute paths for input files. Relative paths are resolved relative to the location of the config.yaml file.
### Image Files

The Image loader supports loading both individual images and batches of images:

1. **Batch Processing**: When the input shape has a batch dimension greater than 1, the loader will process multiple images to form a batch. For example, with a shape of (4, 3, 224, 224), it will load 4 images into a batch.

2. **Automatic Cycling**: If the number of images needed for a batch exceeds the number of available image files, the loader will cycle back to the beginning of the file list.

3. **Format Support**: Supports both NCHW and NHWC formats, automatically handling the necessary transpositions.

4. **Supported File Types**: .jpg, .jpeg, and .png files are supported.

### Pre-processing and Post-processing for Images

The example provides comprehensive pre-processing and post-processing capabilities for image inputs:

#### Pre-processing

Pre-processing is automatically applied to image inputs when the model expects float data. The pre-processing can be configured in the config.yaml file:

```yaml
models:
  model-name:
    pre_process_info:
      input_mean: [123.675, 116.28, 103.53]  # Mean values for RGB channels
      input_scale: [0.017125, 0.017507, 0.017429]  # Scale factors for RGB channels
```

Pre-processing includes:
- Image resizing to match model input dimensions
- Format conversion (RGB to BGR if needed)
- Data type conversion (uint8 to float32)
- Mean subtraction and scaling
- Layout transformation (NHWC to NCHW or vice versa)

#### Post-processing

The example supports task-specific post-processing for common computer vision tasks. Post-processing can be configured in the config.yaml file:

```yaml
models:
  model-name:
    post_process_info:
      task_type: "classification"  # Options: "classification", "detection", "segmentation"
      labels: "path/to/labels.txt"  # Path to labels file for classification
      label_offset: 1  # Offset to apply to label indices
```

Supported post-processing types:
1. **Classification**: Processes model outputs to identify top classes and their probabilities
   - Requires a labels file with class names
   - Supports label offset for models with different indexing

2. **Object Detection**: Processes detection outputs to draw bounding boxes with labels
   - Supports different detection frameworks. Refer [post_process](../utils/post_process/README.md)
   - Handles different output formats automatically

3. **Segmentation**: Processes segmentation masks to create colored visualizations
   - Applies color mapping to segmentation masks
   - Blends the segmentation result with the original image

Post-processing results are saved as:
- Annotated images (.jpg) with visualization of model results
- Text files (.txt) with detailed metadata about the results

This comprehensive pre-processing and post-processing pipeline allows for end-to-end processing of images from input to visualized results, making it easy to work with computer vision models.

### Random

Random loader loads a seeded random value (seed = 0) with specified shape and datatype.

### .NPZ Files

When using .npz files as input data, the following requirements must be met:

1. **Loading by Name (Recommended)**: The NPZ loader will look up each array by the model input name. Name your arrays to match the model input names when creating the file (e.g. `np.savez('data.npz', images=arr0, masks=arr1)`). This is the safest approach for multi-input models as it is order-independent.

   **Index-based Fallback**: If an array key matching the input name is not found, the loader falls back to loading arrays sequentially by position and prints a warning with the available keys. In this case, make sure the arrays are in the same order as the model inputs.

2. **Shape and Data Type**:
   - **Flexible Shape Validation**: The NPZ loader supports more flexible shape validation:
     - Leading dimensions of size 1 are removed before comparison
     - Total volume (product of dimensions) is validated rather than exact shape matching
     - This allows for shape flexibility while ensuring data size compatibility
   - **Data Type**: Each array in the .npz file must have the same data type as expected by the corresponding model input. For example, if the model expects a float32 tensor, the array in the .npz file must be of type np.float32.

### .BIN Files

Binary files provide a way to use raw tensor data as input to models. When using .bin files as input data:

1. **Raw Data Format**: Binary files contain raw tensor data without any metadata. The data should be stored in the exact format expected by the model input.

2. **Shape and Data Type**:
   - The binary data must match the expected shape and size of the model input tensor.
   - The data type should match what the model expects (e.g., float32, int8).
   - No automatic shape or data type conversion is performed.

3. **Usage Scenarios**:
   - Pre-processed data: When you have pre-processed data that's ready for direct model input
   - Testing with specific input patterns: When you want to test model behavior with precisely controlled input values
   - Benchmarking: When you want consistent inputs for performance testing


## Output Files

When running in inference mode, the script saves output image and tensors as binary files in:

- `<script_directory>/outputs/{model_name}/offload/frame_{frame_num}/` (when using TIDL acceleration)
- `<script_directory>/outputs/{model_name}/no_offload/frame_{frame_num}/` (when running on CPU only)

Where:
- `<script_directory>` is the directory containing the basic_example.py script
- `{model_name}` is the name of the model as specified in the config file
- `frame_{frame_num}` is the frame number directory (e.g., "frame_1", "frame_2", etc.)

To collect and save only the first N frames:

```bash
python basic_example.py --infer --dump-frames 1
```


## Verbose Mode

The `--verbose` or `-v` option enables detailed output about the model configuration and its tensors. When this option is enabled, the script will:

1. Display compilation or inference options being used
2. Call the `dump_info()` method of the runtime session after creating it

The `dump_info()` method provides comprehensive information about the model structure:

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
- Analyzing model complexity and memory requirements

## Runtime-Specific Handling

The example handles runtime-specific differences transparently:

### ONNX Runtime:
- Uses ONNXRT class for model handling
- Performs shape inference during compilation
- Uses ONNX-specific tensor details format
- Uses TIDLCompilationProvider/TIDLExecutionProvider for TIDL acceleration
- Supports disabling ONNX Runtime's internal optimization with the `disable_onnx_optimizer`. This option is particularly useful for vision transformer models where the default optimizations might not be beneficial.
- Provides access to ONNX Runtime session options through the public `session_options` property, with defaults for:
  - `log_severity_level = 3` (Warning level logging)
  - `intra_op_num_threads = 1` (Single-threaded execution)
- Session options can be customized before creating import/inference sessions

### TFLite Runtime:
- Uses TFLiteRT class for model handling
- Uses TFLite-specific tensor details format
- Uses TFLite delegates for TIDL acceleration
- Handles tensor resizing when needed

### TIDL Native Runtime:
- Uses TIDLRT class for model handling
- Provides direct access to TIDL's native runtime capabilities
- Includes padding information in tensor details
- Offers maximum performance by bypassing open source runtimes
- Model path is provided in the constructor
- Currently only supports ONNX models

### TVM Runtime:
- Uses TVMRT class for model handling with TIDL acceleration
- Compiles for a specfic platform ('pc', 'evm') or both (default)
- Performs shape inference for ONNX models during compilation
- Bypasses open source runtimes for optimal execution

These differences are handled internally, providing a consistent user experience regardless of the model format being used.
