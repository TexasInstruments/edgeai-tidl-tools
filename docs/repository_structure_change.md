# Repository Structure Changes

This document provides a comprehensive overview of the structural changes made to the EdgeAI TIDL Tools repository from release `11_02_04_00` onwards. These changes were implemented to improve organization, maintainability, and user experience.

> **Note:** The older repository structure is still being maintained for the 11.2 release on the `legacy/rel_11_02` branch. If you prefer to continue using the older structure, you can switch to that branch.

## Table of Contents

- [Motivation](#motivation)
- [Key Structural Changes](#key-structural-changes)
- [Directory Structure Comparison](#directory-structure-comparison)
- [Navigating the New Structure](#navigating-the-new-structure)

## Motivation

The repository structure was redesigned with several goals in mind:

1. **Improved Organization**: Create a more logical and intuitive organization of files and directories
2. **Better Maintainability**: Make the codebase easier to maintain and extend
3. **Enhanced User Experience**: Simplify navigation and usage for both new and experienced users
4. **Clearer Separation of Concerns**: Establish clear boundaries between different components
5. **Consistent API Design**: Provide a more consistent APIs design across different runtimes
6. **Simplified Integration**: Provide abstracted wrapper APIs make it easier to integrate TIDL into your own applications

## Key Structural Changes

### 1. Modular Organization

The new structure organizes code and resources into logical modules:

- **Runtime Components**: All runtime-related code is now organized under the `runtimes/` directory
  - Wrapper APIs are in `runtimes/tidl_wrapper/`
  - Examples are in `runtimes/examples/`
- **Documentation**: Enhanced documentation is centralized in the `docs/` directory with dedicated files for each topic
- **Scripts**: Utility scripts are organized by function in the `scripts/` directory
- **Model Tools**: Model optimization tools are now organized under `osrt-model-tools/`

### 2. Improved API Structure

The API structure has been redesigned to provide:

- Consistent API design across different runtimes (ONNX, TFLite, TVM, TIDL) for both Python and C++
- Better abstraction layers for easier integration
- More comprehensive documentation for each API

### 3. Enhanced Examples

Examples have been reorganized to provide:

- Clearer structure with separate directories for Python and C++ examples
- Consistent configuration approach across examples
- Better documentation and comments
- More comprehensive coverage of use cases

## Directory Structure Comparison

The following table provides a comparison between the previous and new directory structures:

| Component | Previous Structure | New Structure | Description |
|-----------|-------------------|---------------|-------------|
| Wrapper APIs | No implementation of wrapper APIs. Direct usage in examples. | `runtimes/tidl_wrapper/` | Abstracted wrapper APIs for easier integration |
| Examples | Spread in `examples` directory (`osrt_python/ort/onnxrt_ep.py`, `osrt_python/tfl/tflrt_delegate.py`)| `runtimes/examples/` | Simpler and single example to demonstrate the usage of wrapper APIs|
| Documentation | `docs/` directory | `docs/` directory | New Structure introduces more comprehensive documentation organized by topic |
| Setup Scripts | Root-level files (`setup.sh`, `setup_env.sh`, `update_target.sh`) | `scripts/setup/` directory | Better organization of setup scripts |
| Build Scripts | No build scripts | `scripts/build/` directory | Better organization of build scripts |

## Navigating the New Structure

Here's a guide to navigating the new repository structure:

### For Getting Started
1. Start with the top-level [README.md](../README.md) for an overview
2. Follow the setup instructions in the [Getting Started](../README.md#getting-started) section
3. Explore the [User Guide](../README.md#user-guide) section for structured learning

### For Development
1. Explore the wrapper APIs in `runtimes/tidl_wrapper/`
   - [Python Wrapper APIs](../runtimes/tidl_wrapper/python/README.md)
   - [C++ Wrapper APIs](../runtimes/tidl_wrapper/cpp/README.md)
2. Study the examples in `runtimes/examples/`
   - [Python Examples](../runtimes/examples/python/README.md)
   - [C++ Examples](../runtimes/examples/cpp/README.md)
