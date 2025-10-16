# Layer Trace Inspector

A visualization tool for comparing and analyzing layer outputs between reference and test models in TIDL (Texas Instruments Deep Learning) workflows.

## Overview

Layer Trace Inspector is a Streamlit-based interactive web application that helps developers visualize and analyze differences between reference model outputs (e.g., from ONNX) and test model outputs (e.g., from TIDL). It provides various visualization methods to help identify and debug discrepancies between the two models.

## Features

- **Single Layer Analysis**: Compare individual layer outputs between reference and test models
- **Multi-layer Analysis**: Compare multiple intermediate layer outputs simultaneously
- **Network Error Summary**: Generate error summaries across the entire network
- **Interactive Visualizations**: 
  - Side-by-side scatter plots
  - Histograms
  - Error metrics (MAX, AVG, Relative Average)
- **Flexible Data Selection**: Select specific index ranges for detailed analysis
- **Support for Layer Info Mapping**: Map between reference and test layer outputs using a layer info file

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To run the Layer Trace Inspector with basic options:

```bash
streamlit run layer_trace_inspector.py -- --traceRef <path_to_reference_outputs> --traceTest <path_to_test_outputs>
```

### Advanced Usage

To run with layer info mapping:

```bash
streamlit run layer_trace_inspector.py -- --traceRef <path_to_reference_outputs> --traceTest <path_to_test_outputs> --traceInfo <path_to_layer_info_file_from_model_artifacts>
```

### Command Line Arguments

- `--traceRef`: Path to the directory containing reference model output binaries
- `--traceTest`: Path to the directory containing test model output binaries
- `--traceInfo`: (Optional) Path to the layer info file that maps between reference and test layers. Can be found in model artifacts.
- `-v, --verbose`: Enable verbose debug logging

## Web Interface

Once the application is running, you can access it through your web browser. The interface behavior depends on whether you provided a layer info file:

### Without Layer Info File

If you run the tool without providing a layer info file (`--traceInfo`), only a single window will open:

- **SinglePlot**: You'll need to manually select which reference trace and test trace to compare
  - Two dropdown menus will be available to select individual files from each directory
  - You'll need to manually match corresponding traces between reference and test outputs
  - View side-by-side scatter plots and histograms
  - Analyze error metrics
  - Optionally specify start and end indices for detailed analysis

### With Layer Info File

If you provide a layer info file, the interface offers multiple tabs:

#### SinglePlot

- Select a specific layer to analyze from a dropdown that uses the mapping from the layer info file
- The corresponding reference and test traces are automatically paired
- View side-by-side scatter plots and histograms
- Analyze error metrics
- Optionally specify start and end indices for detailed analysis

#### Network Error Summary

- View error metrics across all layers in the network
- Identify layers with high error rates
- Compare different error metrics (Relative MAE, MAX, Absolute MAE)

#### Multi Plot

- Select multiple layers to analyze simultaneously
- Compare error patterns across different layers
- Identify systematic issues affecting multiple layers

## Output Binary Format

The tool expects output binaries to be in raw float32 format. Each binary file should contain the flattened output tensor of a single layer.

## Troubleshooting

- Ensure that reference and test output directories contain valid binary files
- Check that the layer info file (if provided) correctly maps between reference and test layers
- Verify that the binary files are in the expected format (raw float32)
- If you encounter memory issues with large models, try analyzing specific layers or index ranges instead of the entire model

## License

Copyright (c) 2026, Texas Instruments Incorporated. All rights reserved.
