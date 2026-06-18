# Frequently Asked Questions (FAQ)

This document provides answers to frequently asked questions about using the EdgeAI TIDL Tools.

## Table of Contents

- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
  - [Table of Contents](#table-of-contents)
  - [General Questions](#general-questions)
    - [What is TIDL?](#what-is-tidl)
    - [Which TI devices are supported?](#which-ti-devices-are-supported)
    - [What frameworks are supported?](#what-frameworks-are-supported)
  - [Setup and Installation](#setup-and-installation)
    - [How do I set up the development environment?](#how-do-i-set-up-the-development-environment)
    - [How do I set up an NFS server on my PC for mounting to a TI SOC?](#how-do-i-set-up-an-nfs-server-on-my-pc-for-mounting-to-a-ti-soc)
    - [How do I check if my setup is correct?](#how-do-i-check-if-my-setup-is-correct)
  - [Model Compilation](#model-compilation)
    - [Why can't I compile models on the TI SOC?](#why-cant-i-compile-models-on-the-ti-soc)
    - [How do I know if my model is fully offloaded?](#how-do-i-know-if-my-model-is-fully-offloaded)
  - [Model Optimization Tools](#model-optimization-tools)
    - [Why should I optimize my models before using TIDL?](#why-should-i-optimize-my-models-before-using-tidl)
    - [What is tidl-onnx-model-optimizer?](#what-is-tidl-onnx-model-optimizer)
    - [What can I do with tidl-onnx-model-optimizer?](#what-can-i-do-with-tidl-onnx-model-optimizer)
    - [What is osrt-model-tools?](#what-is-osrt-model-tools)
      - [ONNX Tools:](#onnx-tools)
      - [TFLite Tools:](#tflite-tools)
  - [Backward Compatibility](#backward-compatibility)
    - [How do I check SDK version compatibility?](#how-do-i-check-sdk-version-compatibility)
    - [What is the update\_target script?](#what-is-the-update_target-script)
  - [Basic Issues and Debugging](#basic-issues-and-debugging)
    - [Linker Failure](#linker-failure)
    - [Debugging](#debugging)

## General Questions

### What is TIDL?

TIDL (TI Deep Learning) is a comprehensive software product for acceleration of Deep Neural Networks (DNNs) on TI's embedded devices. It supports heterogeneous execution of DNNs across Cortex-A based MPUs, TI's latest generation C7x DSP, and TI's DNN accelerator (MMA).

### Which TI devices are supported?

TIDL supports various TI devices including:
- AM62A
- J722S \| TDA4AEN \| AM67A
- J721E \| TDA4VM
- J721S2 \| TDA4VL \| AM68A
- J784S4 \| TDA4VH \| AM69A
- AM62 (without hardware acceleration)

For the most up-to-date information, refer to the [Supported Devices and Compatibility](../README.md#supported-devices-and-compatibility) section in the main README.

### What frameworks are supported?

TIDL supports multiple frameworks for model inference:
- ONNX Runtime
- TensorFlow Lite Runtime
- TVM Runtime
- TIDL Runtime (native)

## Setup and Installation

### How do I set up the development environment?

For detailed setup instructions, refer to the [Setup on X86 PC](../README.md#setup-on-x86-pc) section in the main README. In general, you need to:

1. Install system dependencies
2. Clone the repository
3. Set up a Python environment (virtual environment or Docker)
4. Download and install tools and dependencies
5. Set up environment variables
6. Build C++ components (if needed)

### How do I set up an NFS server on my PC for mounting to a TI SOC?

Setting up an NFS server on your PC allows you to easily share files with your TI SOC, including compiled model artifacts. Here's how to set it up:

**On the X86 PC (NFS Server):**

1. Install NFS server:
   ```bash
   sudo apt-get update
   sudo apt-get install nfs-kernel-server
   ```

2. Create an export directory or use your existing edgeai-tidl-tools directory:
   ```bash
   # Note the path to your edgeai-tidl-tools directory
   EXPORT_DIR=/path/to/edgeai-tidl-tools
   ```

3. Configure NFS exports:
   ```bash
   sudo bash -c "echo '$EXPORT_DIR *(rw,sync,no_subtree_check,no_root_squash)' >> /etc/exports"
   ```

4. Restart the NFS server:
   ```bash
   sudo exportfs -a
   sudo systemctl restart nfs-kernel-server
   ```

**On the TI SOC (NFS Client):**

1. Create a mount point:
   ```bash
   mkdir -p /mnt/edgeai-tidl-tools
   ```

2. Mount the NFS share:
   ```bash
   # Replace X86_PC_IP with the actual IP address of your X86 PC
   mount <X86_PC_IP>:/path/to/edgeai-tidl-tools /mnt/edgeai-tidl-tools
   ```

### How do I check if my setup is correct?

You can verify your setup by running the basic examples:

1. For Python:
   ```bash
   python3 ./runtimes/examples/python/basic_example/basic_example.py --compile
   python3 ./runtimes/examples/python/basic_example/basic_example.py --infer
   ```

2. For C++:
   ```bash
   ./runtimes/examples/cpp/build/basic_example/basic_example
   ```

If these examples run without errors, your setup is correct.

## Model Compilation

### Why can't I compile models on the TI SOC?

Model compilation is only supported on x86 PCs because:

1. The compilation process requires significant computational resources that may not be available on the TI SOC.
2. The compilation tools are designed to run on x86 architecture.
3. The workflow is designed for development on a PC and deployment on the TI SOC.

You need to compile your models on an x86 PC and then transfer the compiled artifacts to the TI SOC for inference.

### How do I know if my model is fully offloaded?

A model is considered "fully offloaded" when all of its operations can be executed on the TIDL hardware accelerators without requiring CPU involvement. You can determine if your model is fully offloaded by:

1. Examining the compilation log for messages indicating full offload.
2. Checking the model artifacts folder - a fully offloaded model will have just a singular network and IO binary file.
3. Looking at the `tempDir/runtimes_visualization.svg` file - a fully offloaded model will have all nodes encapsulated under a single colored block.

## Model Optimization Tools

### Why should I optimize my models before using TIDL?

Many general optimizations and tricks can be performed on models offline even before they are given to TIDL. These first-level optimizations can help in multiple ways:

1. **Improved TIDL Compatibility**: Certain model optimizations can allow layers to be better supported by TIDL by keeping layer constraints in mind.
2. **Better Performance**: Pre-optimized models often execute faster on TIDL hardware.
3. **Simplified Input/Output Processing**: Tools like RGB to YUV converters can simplify the integration with camera pipelines.

The tidl-onnx-model-optimizer package provides utilities to perform these optimizations easily and effectively.

### What is tidl-onnx-model-optimizer?

OSRT Model Tools is a collection of functions for optimizing and modifying ONNX models to improve their compatibility and performance with TIDL (TI Deep Learning). It provides tools to help prepare models for efficient execution on TI devices.

### What can I do with tidl-onnx-model-optimizer?

The tidl-onnx-model-optimizer package provides several functions for model optimization:
- **Model Optimization**: Optimize ONNX models for TIDL inference by performing various user selected optimizations


### What is osrt-model-tools?
osrt-model-tools contains a set of utilities to do the following:

#### ONNX Tools:
- **Model Input Pre-Processing**: Modify ONNX models input to add pre-processing as part of the model
- **RGB to YUV Conversion**: Convert RGB-trained models to accept YUV (NV12) image format as input
- **Batch Size Modification**: Update models to support specific batch dimensions
- **Intermediate Outputs**: Add output layers to all nodes for debugging
- **Model Extraction**: Extract subgraphs from models for targeted optimization
- **Node Name Management**: Get nodes between specific layers or simplify intermediate tensor names

#### TFLite Tools:
- **Model Pre-Processing**: Modify TFLite models input to add pre-processing as part of the model
- **RGB to YUV Conversion**: Convert RGB-trained TFLite models to accept YUV (NV12) input format

For detailed documentation on each tool, refer to:
- [ONNX Tools Documentation](../model-tools//osrt-model-tools/osrt_model_tools/onnx_tools)
- [TFLite Tools Documentation](../model-tools/osrt-model-tools/osrt_model_tools/tflite_tools)

## Backward Compatibility

### How do I check SDK version compatibility?

To check if your EdgeAI TIDL Tools version is compatible with your SDK version:

1. Refer to the [SDK Version Compatibility Table](./sdk_version_compatibility_table.md) document.
2. Find your SDK version in the appropriate section (e.g., SDK Version 11.01.xx.xx).
3. Check the listed EdgeAI TIDL Tools versions that are compatible with your SDK.
4. Note the compatibility type:
   - **Default**: Standard release for the specified SDK version.
   - **Patch with default compatibility**: Includes additional features and fixes for the corresponding default SDK version.
   - **Patch with backward compatibility**: Includes additional features and fixes designed to work on older SDK versions.

### What is the update_target script?

The [update_target.sh](../scripts/setup/update_target.sh) script is used to update a target device with firmware, components, and libraries from a newer TIDL tools version while using a previous SDK version, enabling backward compatibility.

The `update_target.sh` script updates two main component groups:

1. OSRT (Open Standard Runtime) components wheels, libraries and headers
2. TIDL Firmware and Libraries

The `update_target.sh` script is a simple bash script that serves as a reference for manually updating components.

## Basic Issues and Debugging

### Linker Failure

`libvx_tidl_rt` library is used as a bridge between the TIDL runtime firmware and the user applications. It provides APIs which can be called in ARM applications, which internally takes care of the delegation to C7x core for inference.
This is part of the 'edgeai-tidl-tools' offering.

Users might want to directly load this shared library to use in their application.
When building a basic application with `libvx_tidl_rt`, user may encounter a linker failure such as:

```
lib/libvx_tidl_rt.so: undefined reference to `process_hwaop_imm(int)'
```

This symbol exists but is undefined within `libvx_tidl_rt.so`. This can be avoided by including the following LD flag during linking:

```
-Wl,-unresolved-symbols=ignore-in-shared-libs
```

### Debugging

[Debugging](./debugging.md) provides information on some common and frequently run into issues and errors and also basic debugging of model compilation. We highly recomment you check out this page for more detailed. Few points that debugging page covers are

- Setup , Permission and build errors.
- Debug trace logs
- Debugging Model Compilation
- Debugging Model Inference - Model Artifacts Incompatibility, Inferencing Failures, Incorrect Inference Results, Inference Performance
