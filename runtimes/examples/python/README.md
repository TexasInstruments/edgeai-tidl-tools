# Python Examples

This directory contains Python examples demonstrating how to use the [Python wrapper](../../tidl_wrapper/python/README.md) libraries for model compilation and inference with deep learning models on Texas Instruments devices.

## Available Examples

### Basic Example

The [Basic Example](./basic_example/README.md) demonstrates how to use ONNXRT, TFLiteRT, and TVMRT wrapper modules for model compilation and inference with TIDL acceleration. It provides a command-line interface to run model compilation and inference with various configuration options defined in config.yaml.

For detailed documentation, usage instructions, and command-line options, see the [Basic Example README](./basic_example/README.md).

## Utility Libraries

The [utils](./utils) directory contains utility functions and classes used by the examples:

- **dataset_loader**: Input data loading utilities for random data, image, .npz files, and .bin files

These utilities provide common functionality that can be reused across different examples.
