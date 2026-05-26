# TVMRT Wrapper Module

The TVMRT wrapper module provides a simplified interface for working with TVM Runtime with TIDL (Texas Instruments Deep Learning) acceleration. This module enables easy model import and inference with TIDL offloading capabilities.

For more details on usage of TVM, please refer to this link - https://software-dl.ti.com/codegen/docs/tvm/tvm_tidl_users_guide/

## Overview

The `TVMRT` class encapsulates the functionality needed to:

- Import models with TVM runtime
- Run inference on models with TIDL acceleration
- Collect performance metrics for model execution
- Display detailed model and tensor information

This module is designed to simplify the process of using TVM Runtime with TIDL acceleration by providing a clean, high-level API.

## Requirements

- Python 3.10
- TVM Runtime
- ONNX Runtime (for ONNX models)
- NumPy
- TIDL Tools (for model compilation)

## Note

For inference on target machine, user needs to manually install additional dependencies using the below command:
``` python
pip3 install psutil typing_extensions
```

## API Documentation

### TVMRT Class

```python
class TVMRT:
    """
    This class provides simple APIs to import/infer models for TIDL using
    TVM Runtime interface
    """
```

#### Constructor

```python
def __init__(self, model_path : str, tidl_offload : bool = True, target_machine : list = ['pc', 'evm']):
    """
    Initializes a new TVMRT object.

    Args:
        model_path (str): Path to onnx model.
        tidl_offload (bool): Optional argument to enable/disable TIDL offload. Default: True.
        target_machine (list): Platfrom to compile the model for e.g. 'pc', 'evm'

    Note:
        - When both 'pc' and 'evm' are specified, compilation runs for both platforms
        - Artifacts are generated with platform-specific suffixes (.pc, .evm)
        - For inference, the appropriate platform is automatically selected based on runtime architecture
    """
```

#### Methods

##### create_import

```python
def create_import(self, options: dict = None):
    """
    Create import session for model compilation.

    Args:
        options (dict): Dictionary containing import options
    """
```

##### run_import

```python
def run_import(self, input: dict, output_keys: list = None):
    """
    Run import for model compilation.

    Args:
        input (dict): Input data dictionary in {'input_name': input_data} format
        output_keys (list): Optional list to filter output based on output name

    Returns:
        dict: Output in {'output_name': output_data}
    """
```

##### create_infer

```python
def create_infer(self, options: dict = None):
    """
    Create inference session.

    Args:
        options (dict): Dictionary containing inference options
    """
```

##### run_infer

```python
def run_infer(self, input: dict, output_keys: list = None):
    """
    Run inference.

    Args:
        input (dict): Input data dictionary in {'input_name': input_data} format
        output_keys (list): Optional list to filter output based on output name

    Returns:
        dict: Output in {'output_name': output_data}
    """
```

##### get_performance

```python
def get_performance(self):
    """
    This method returns performance data.

    Returns:
        dict: A dictionary containing performance metrics with values and units.
              'total_time': (value, "ms") - Total time taken for run
              'core_time': (value, "ms") - Total time taken barring the IO copy time
              'subgraph_time': (value, "ms") - Total TIDL Subgraphs processing time
              'read_total': (value, "bytes") - Total DDR Read bytes [X for x86 runs]
              'write_total': (value, "bytes") - Total DDR Write bytes [X for x86 runs]
              'total': (value, "bytes") - Total DDR Read+Write bytes [X for x86 runs]
              'num_subgraphs': (value, "") - Total Detected subgraphs
    """
```

##### dump_info

```python
def dump_info(self):
    """
    Prints detailed information about the model and its tensors.

    This method displays model path, input/output tensor counts, and detailed
    information about each tensor including name, type, shape, and size.

    This is particularly useful for debugging and understanding the model structure
    when used with the verbose mode in applications.
    """
```

## Example Usage

For a complete working example, please refer to the basic example in [runtimes/examples/python/basic_example](../../../examples/python/basic_example/README.md)

> [NOTE] While running on x86, make sure to have TIDL_TOOLS_PATH set and add it to LD_LIBRARY_PATH environment variable as well.

### Model Compilation Example

```python
from tvmrt_wrapper import TVMRT
import numpy as np
import os

# Initialize TVMRT with model path
model_path = "path/to/model.onnx"
artifacts_path = "path/to/artifacts"
tidl_tools_path = os.environ.get("TIDL_TOOLS_PATH")

# Create session for compilation
session = TVMRT(model_path=model_path, tidl_offload=True, target_machine=['pc', 'evm'])

# Define compilation options
compile_options = {
    "artifacts_folder": artifacts_path,
    "tidl_tools_path": tidl_tools_path,
    "tensor_bits": 8,                          # 8-bit quantization
    "accuracy_level": 1,
    "advanced_options:calibration_frames": 10
}

# Create import session
session.create_import(options=compile_options)

# Optional: Display detailed model information
session.dump_info()

num_frames = compile_options["advanced_options:calibration_frames"]

for i in range(num_frames):
    # Prepare input data for calibration
    input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
    input_dict = {"input": input_data}  # Replace "input" with your model's input name

    # Run import (compilation)
    output = session.run_import(input=input_dict)
print("Compilation completed")
```

### Inference Example

```python
from tvmrt_wrapper import TVMRT
import numpy as np

# Initialize TVMRT with model path
model_path = "path/to/model.onnx"
artifacts_path = "path/to/artifacts"

# Create session for inference
session = TVMRT(model_path=model_path, tidl_offload=True)

# Define inference options
infer_options = {
    "artifacts_folder": artifacts_path
}

# Create inference session
session.create_infer(options=infer_options)

# Optional: Display detailed model information
session.dump_info()

num_frames = 5

for i in range(num_frames):
    # Prepare input data
    input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
    input_dict = {"input": input_data}  # Replace "input" with your model's input name

    # Run inference
    output = session.run_infer(input=input_dict)

    # Get performance metrics
    performance = session.get_performance()
    print(performance)

print("Inference completed")
```
## TVMC Integration

```python
# To enable TVMC mode, set the below environment variable
export USE_TVMC_MODE=1
```
When USE_TVMC_MODE is set, TVMRT uses the TVM Compiler (TVMC) command-line interface internally for the compilation.
- TVMC requires calibration data in form of npz file, which is created in this application from the input calibration data.
- For more details on TVMC, refer TVMC documentation in TVM docs.

## Compilation Options

When using the `create_import` method for model compilation, you need to provide compilation options to configure the TIDL compilation process. Here are the key options:

- `artifacts_folder`: (Required) Path to store model artifacts generated during compilation
- `tidl_tools_path`: (Required) Path to TIDL tools installation

Refer to [Model Compilation Options](../../../../docs/model_compilation.md#compilation-options) for details about available compilation options.

## Inference Options

When using the `create_infer` method for model inference, you need to provide inference options to configure the TIDL inference process. Here are the key options:

- `artifacts_folder`: (Required) Path to the artifacts folder containing compiled model artifacts

Refer to [Model Inference Options](../../../../docs/model_inference.md#inference-options) for details about available inference options.

## Performance Metrics

The `get_performance()` method returns a dictionary with the following metrics, where each value is a tuple containing the metric value and its unit:

| Metric | Description | Unit |
|--------|-------------|------|
| `total_time` | Total time taken for model inference | milliseconds (ms) |
| `core_time` | Processing time excluding I/O copy operations | milliseconds (ms) |
| `subgraph_time` | Time spent in all TIDL subgraph execution | milliseconds (ms) |
| `read_total` | Total DDR read bytes | bytes |
| `write_total` | Total DDR write bytes | bytes |
| `total` | Total DDR Read+Write bytes | bytes |
| `num_subgraphs` | Total Subgraphs detected | |

## Model Information

The `dump_info()` method provides detailed information about the model and its tensors, including:

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

Example output:
```
Model Path        = /path/to/model.onnx
Number of Inputs  = 1
INPUT [0]:
  Name     = input
  Type     = float32
  Shape    = [1, 3, 224, 224]
  Num Dims = 4
  Num Elem = 150528
Number of Outputs = 1
OUTPUT [0]:
  Name     = output
  Type     = float32
  Shape    = [1, 1000]
  Num Dims = 2
  Num Elem = 1000
