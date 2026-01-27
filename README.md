# TIDL - TI Deep Learning Product
TIDL is a comprehensive software product for acceleration of Deep Neural Networks (DNNs) on TI's embedded devices. It supports heterogeneous execution of DNNs across cortex-A based MPUs, TI's latest generation C7x DSP and TI's DNN accelerator (MMA). TIDL is released as part of TI's Software Development Kit (SDK) along with additional computer vision functions and optimized libraries including OpenCV. TIDL is available on a variety of embedded devices from Texas Instruments.

TIDL is a fundamental software component of [**TI's Edge AI solution**](https://www.ti.com/edgeai).
TI's Edge AI solution simplifies the whole product life cycle of DNN development and deployment by providing a rich set of tools and optimized libraries. DNN-based product development requires two main streams of expertise:
* **Data Scientists**, who can design and train DNNs for targeted applications
* **Embedded System Engineers**, who can design and develop inference solutions for real time execution of DNNs on low power embedded devices


TI's Edge AI solution provides the right set of tools for both of these categories:

* [**Edge AI Studio**](https://dev.ti.com/edgeai/): Integrated development environment for development of AI applications for edge processors, hosting tools like **Model Composer** to train, compile and deploy models with a click of a mouse button and **Model Analyzer** to let you evaluate and analyze deep learning model performance on TI devices from your browser in minutes
* [**Model zoo**](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modelzoo): A large collection of pre-trained models for data scientists, which along with TI's Model Selection Tool enables picking the ideal model for TI's embedded devices
* [**Training and quantization tools**](https://github.com/TexasInstruments/edgeai) for popular frameworks, allowing data scientists to make DNNs more suitable for TI devices
* [**Edge AI Benchmark**](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-benchmark): A Python-based framework which allows you to perform accuracy and performance benchmarks. Accuracy benchmarks can be performed without a development board, but for performance benchmarks, a development board is needed.
* [**Edge AI TIDL Tools**](#edgeai-tidl-tools): Edge AI TIDL Tools provided in this repository shall be used for model compilation on X86. Artifacts from the compilation process can be used for model inference. Model inference can happen on an X86 machine (host emulation mode) or on a development board with TI SOC. This repository provides simple-to-use runtime wrapper APIs and examples which are intended to serve as a ready-to-use tool for model compilation and inference as well as a reference for users to develop their own applications for TIDL. 

The figure below illustrates the workflow of DNN development and deployment on TI devices:
<div align="center">
<img src="./docs/assets/dnn-workflow.png" alt="TI'S Edge AI workflow">
</div>

# EdgeAI TIDL Tools

<!-- TOC -->
   - [Introduction](#introduction)
   - [What does edgeai-tidl-tools provide?](#what-does-edgeai-tidl-tools-provide)
   - [Supported Devices and Compatibility](#supported-devices-and-compatibility)
   - [Current Version Compatibility](#current-version-compatibility)
   - [Workflow](#workflow)
      - [Python Workflow](#python-workflow)
      - [C++ Workflow](#c-workflow)
      - [Cross-Language Workflow](#cross-language-workflow)
   - [Runtime Wrapper APIs and Examples](#runtime-wrapper-apis-and-examples)
   - [Getting Started](#getting-started)
      - [Setup on X86 PC](#setup-on-x86-pc)
      - [Running on X86 PC](#running-on-x86-pc)
      - [Setup on TI SOC](#setup-on-ti-soc)
      - [Running on TI SOC](#running-on-ti-soc)
   - [User Guide](#user-guide)
   - [Test Reports](#test-reports)
   - [Additional Support](#additional-support)
   - [License](#license)
<!-- /TOC -->

<br>
<br>

> **Note:** This repository features a completely redesigned structure from release `11_02_04_00` onwards compared to previous versions of edgeai-tidl-tools. For details on the structural changes, migration guidance, and how to still access the older repository structure, please refer to the [Repository Structure Changes](./docs/repository_structure_change.md) documentation.

## Introduction
TIDL provides multiple deployment options with industry defined inference engines as well as TIDL's own native inference engine:

### Execution via Open Source Runtimes

* **ONNX Runtime**: ONNX Runtime based inference with execution on Cortex-A + C7x-MMA, using TIDL as execution provider.
* **TFLite Runtime**: TensorFlow Lite based inference with execution on Cortex-A + C7x-MMA, using TIDL as delegate.
* **TVM Runtime**: TVM based inference with execution on Cortex-A + C7x-MMA

   These enable:
   1. Open Source Runtime (OSRT) as the top level inference for user applications
   2. Offloading subgraphs to C7x-MMA for accelerated execution with TIDL
   3. Running optimized code on ARM core for layers that are not supported by TIDL

### Native execution

* **TIDL-RT / TIDL Runtime**: TIDL's native runtime provides direct execution on C7x-MMA without using any open source runtime.

   TIDL Runtime offers:
   1. Complete execution of all the layers on the C7x-MMA without ARM core involvement in the execution path
   2. Optimized for maximum performance on TI's C7x DSP and MMA
   3. Suitable for performance and safety critical applications

> **Note on Terminology**: In the documentation and APIs, you may encounter both **"TIDL-RT"** and **"TIDLRUNTIME"** terms. TIDL-RT refers to the underlying C APIs that provide native runtime functionality. The open-source runtimes directly interface with these C APIs via execution provider / delegate mechanism. TIDLRUNTIME specifically refers to the Python bindings built on top of the TIDL-RT C APIs, providing a Python interface to the same functionality.

For more information, check [TIDLRT](./docs/tidlrt.md)


## What does edgeai-tidl-tools provide?

This repository is meant to serve as a reference application allowing users to draw inspiration to create their own applications around TIDL.

This repository provides:

- Support for both x86 and TI SOC.
- Tools for model compilation on x86 system. This will be downloaded under `tools` folder as a part of [Setup on X86 PC](#setup-on-x86-pc)
- Ready-to-use [Python and C++ Wrapper APIs](./runtimes/tidl_wrapper/README.md) for easy integration
- Ready-to-use [Python and C++ reference](./runtimes/examples/README.md) examples.
- Comprehensive [userguide](#user-guide) including workflow, compilation and inference processes, supported operators, debugging etc.
- Utility scripts for optimizing and modifying models using [osrt-model-tools](./osrt-model-tools/README.md)
- Utility scripts for setup, build, docker and debugging.

<div align="center">
<img src="./docs/assets/edgeai_tidl_tools_components.png" width="800px">
</div>

This modular architecture allows for flexible usage patterns:
1. Use the entire stack for end-to-end model compilation and inference
2. Use only the wrapper APIs to build custom applications
3. Mix and match Python and C++ components as needed

## Supported Devices and Compatibility
<div align="center">

| Device Family (Product) | Hardware Acceleration | ONNX Runtime | TFLite Runtime | TVM Runtime | TIDL Runtime |
|-------------------------|:--------------------:|:------------:|:-------------:|:-----------:|:------------:|
| AM62A                   | :heavy_check_mark:   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| J722S \| TDA4AEN \| AM67A | :heavy_check_mark:   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| J721E \| TDA4VM         | :heavy_check_mark:   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| J721S2 \| TDA4VL \| AM68A | :heavy_check_mark:   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| J784S4 \| TDA4VH \| AM69A | :heavy_check_mark:   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| AM62                    | :x:                  | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: |
</div>

## Version Compatibility

This section provides information about the compatibility between the **current branch/tag** of EdgeAI TIDL Tools and the SDK versions as well as the runtime versions

<div align="center">

| Device Family | SDK Version |
|---------------|-------------|
| AM62A         | N/A |
| J722S \| TDA4AEN \| AM67A | N/A |
| J721E \| TDA4VM | [Processor SDK RTOS 11.02.00.06](https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E/11.02.00.06)<br>[Processor SDK LINUX 11.02.00.04](https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-J721E/11.02.00.04) |
| J721S2 \| TDA4VL \| AM68A | [Processor SDK RTOS 11.02.00.06](https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721S2/11.02.00.06)<br>[Processor SDK LINUX 11.02.00.04](https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-J721S2/11.02.00.04) |
| J784S4 \| TDA4VH \| AM69A | [Processor SDK RTOS 11.02.00.06](https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J784S4/11.02.00.06)<br>[Processor SDK LINUX 11.02.00.04](https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-J784S4/11.02.00.04) |
| AM62          | N/A |

</div>
<br>
<br>
<div align="center">

| Runtime | Version | Supported Devices |
|---------|---------|-------------------|
| ONNX Runtime | 1.15.0 | All devices |
| TFLite Runtime | 2.12.0 | All devices |
| TVM Runtime | 0.18.0 | All devices |

</div>


### Notes:
- For detailed compatibility information and backward compatibility notes, please refer to the [SDK Version Compatibility Table](docs/sdk_version_compatibility_table.md).
- When using this branch/tag, make sure to compile your models with the appropriate settings for your target device.
- Runtime versions are fixed for a given branch/tag of EdgeAI TIDL Tools.

## Workflow
The typical workflow involves two main steps:

1. **Model Compilation** (Python)
   - Convert and optimize models for TIDL acceleration
   - Generate model artifacts that can be used for inference

2. **Model Inference** (Python or C++)
   - Load the compiled model artifacts
   - Run inference with TIDL acceleration
   - Process the model outputs

<div align="center">

<table>
   <tr>
      <td style="vertical-align: top; padding-right: 180px">

| Operation | X86_PC | TI SOC |
|-----------|:------:|:------:|
| Model Compilation | :heavy_check_mark: | :x: |
| Model Inference | :heavy_check_mark: | :heavy_check_mark: |
   </td>

   <td style="vertical-align: top;">

| Operation | Python API | C++ API |
|-----------|:------:|:------:|
| Model Compilation | :heavy_check_mark: | :x: |
| Model Inference | :heavy_check_mark: | :heavy_check_mark: |
   </td>
  </tr>
</table>

</div>


<div align="center">
<p float="left">
  <img src="./docs/assets/onnxrt-workflow.png" width="650" />
  <img src="./docs/assets/tflitert-workflow.png" width="650" /> 
</p>
</div>

This workflow allows for flexible development and deployment scenarios:

1. Use Python for model development, compilation, and prototyping
2. Use C++ for production deployment with optimized performance
3. Share the same compiled model artifacts between Python and C++ implementations

> **Note on Runtime Compatibility**:
> - For tidlrt, the artifacts are agnostic of the runtime used during compilation and can run any model artifacts as long as all nodes are offloaded.
> - Models compiled via onnxrt can only be used for inference with onnxrt and tidlrt
> - Models compiled via tflitert can only be used for inference with tflitert and tidlrt

## Runtime Wrapper APIs and Examples

The EdgeAI TIDL Tools are structured with two main components under [./runtimes](./runtimes/):

1. [**Wrapper APIs**](./runtimes/tidl_wrapper/README.md): These provide an abstraction layer that encapsulates the intricacies of different runtimes (ONNX Runtime, TFLite, TVM, TIDL) and offer a common set of APIs for users.
2. [**Examples**](./runtimes/examples/README.md): These serve two purposes, as reference applications demonstrating how to use the wrapper APIs and as ready-to-use applications for basic use cases.

This architecture allows you to either use the provided examples application directly or develop your own applications using the wrapper APIs as a foundation.

For more detailed information about the wrapper APIs and examples, refer to the documentation:
- [Python Wrapper APIs](runtimes/tidl_wrapper/python/README.md)
- [C++ Wrapper APIs](runtimes/tidl_wrapper/cpp/README.md)
- [Python examples](runtimes/examples/python/README.md)
- [C++ examples](runtimes/examples/cpp/README.md)

## Getting Started

This section provides instructions to get started with the EdgeAI TIDL Tools.

### Setup on X86 PC

1. **Install system dependencies**:
   ```bash
   sudo apt-get install libyaml-cpp-dev libglib2.0-dev python-setuptools
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
   cd edgeai-tidl-tools
   git checkout <TAG Compatible with SDK version>
   ```
3. **Setup development environment**

   We recommend using either a virtual Python environment or Docker for a clean and isolated setup.

   **Option 1: Virtual Python Environment** : Create and activate python 3.10 virtual environment

   **Option 2: Docker** : For details, see the [Docker README](scripts/docker/README.md).

3. **Download and install tools and dependencies**:
   ```bash
   ./scripts/setup/setup.sh
   ```

4. **Setup environment variables**:
    ```bash
    source ./scripts/setup/setup_env.sh <SOC> #Ex: J721S2 or AM62A or ... refer to Device Family in Supported Devices above
    ```
    For more details, see the [Setup README](scripts/setup/README.md).

5. **Build C++ components** (Only if you want to use C++ APIs and example):
   ```bash
   ./scripts/build/build_cpp.sh --clean
   ./scripts/build/build_cpp.sh
   ```
   For more details, see the [Build README](scripts/build/README.md).


### Running on X86 PC

> **Important:** Before running any examples, ensure that you have properly set up the environment variables by sourcing the `scripts/setup/setup_env.sh`. Missing environment variables are a common cause of errors.

This repository provides out-of-box examples to validate your setup and demonstrate how to use the TIDL tools.

1. **Running Python Examples**:

   The Python examples demonstrate how to use the various runtime libraries for model compilation and inference with deep learning models.

   ```bash
   cd ./runtimes/examples/python/basic_example/

   # Running onnxrt examples
   python3 basic_example.py --config ./config.yaml -r onnxrt --compile
   python3 basic_example.py --config ./config.yaml -r onnxrt --infer

   # Running tflitert examples
   python3 basic_example.py --config ./config.yaml -r tflitert --compile
   python3 basic_example.py --config ./config.yaml -r tflitert --infer

   # Running tvmrt examples
   python3 basic_example.py --config ./config.yaml -r tvmrt --compile
   python3 basic_example.py --config ./config.yaml -r tvmrt --infer

   cd -
   ```

   - Model artifacts will be saved in `./runtimes/examples/model-artifacts/`
   - Outputs will be saved in `./runtimes/examples/python/basic_example/outputs/{model_name}/offload/frame_{frame_num}/`

2. **Running C++ Examples**:

   The C++ examples demonstrate how to use the various runtime libraries for inference with deep learning models. C++ currently does not support model compilation.

   ```bash
   cd ./runtimes/examples/cpp/basic_example
   ../bin/Release/basic_example --config ./config.yaml
   cd -
   ```

   - Outputs will be saved in `./runtimes/examples/cpp/basic_example/outputs/{model_name}/offload/frame_{frame_num}/`

Running these examples successfully validates that your setup is correct. If you encounter any issues running these examples, please refer to the [debugging guide](docs/debugging.md) for troubleshooting steps.


### Setup on TI SOC

All the libraries and dependencies needed for TI SOC should already be packaged as part of the SDK. 

> [NOTE] Ensure that your TI SOC is supported as listed in the [Supported Devices and Runtimes](#supported-devices-and-runtimes) table above. Additionally, check [Current Version Compatibility](#current-version-compatibility) to check the supported SDK version.

You have two options for setting up the EdgeAI TIDL Tools repository on your TI SOC:

1. **Clone/Mount the repository on the SOC**:
   You can either clone the repository or mount it from x86 PC.
   ```bash
   # Cloning
   git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
   cd edgeai-tidl-tools
   git checkout <TAG Compatible with SDK version>

   # OR

   # Mount using nfs.
   # Make sure /path/to/edgeai-tidl-tools on X86 PC is NFS mountable
   mkdir -p /mnt/edgeai-tidl-tools
   mount <X86_PC_IP>:/path/to/edgeai-tidl-tools /mnt/edgeai-tidl-tools
   cd /mnt/edgeai-tidl-tools
   ```

   Check [FAQ](docs/faq.md#how-do-i-set-up-an-nfs-server-on-my-pc-for-mounting-to-a-ti-soc) for details on how to setup NFS server on x86 PC.
2. **Setup environment variables**:
   ```bash
   export SOC=<SOC> #Ex: J721S2 or AM62A or ... refer to Device Family in Supported Devices above
   ```

3. **Build C++ components** (Only if you want to use C++ APIs and example):
   
   You have two options for building C++ components on TI SOC:

   > **Important Note for SDK 11.2:** The yaml-cpp library is not packaged in SDK 11.2. This will be fixed in future SDK releases. As a result:
   > - Cross-compilation with 11.2 SDK will not work for compiling the C++ examples
   > - For native compilation, you must first clone, build, and install yaml-cpp on the SoC as a prerequisite

   **Option 1: Native Compilation** (Building directly on the TI SOC):
   ```bash
   # For SDK 11.2, first install yaml-cpp
   git clone -b 0.8.0 https://github.com/jbeder/yaml-cpp.git
   cd yaml-cpp
   mkdir build && cd build
   cmake .. -DYAML_BUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr
   make -j2
   make install
   cd ../../
   rm -rf yaml-cpp

   # Then build the C++ components
   ./scripts/build/build_cpp.sh --clean
   ./scripts/build/build_cpp.sh
   ```
   For more details, see the [Build Readme](scripts/build/README.md).

   **Option 2: Cross-Compilation** (Building on X86 PC for TI SOC):
   ```bash
   # Following steps are executed on x86 PC to cross-compile for aarch64 
   # Note: This will not work with SDK 11.2 due to missing yaml-cpp package
   export SDK_PATH=<path to sdk> # Ex: /home/user/ti-processor-sdk-rtos-j784s4-evm-11_02_00_05
   export TARGET_CPU=aarch64
   ./scripts/build/build_cpp.sh --clean
   ./scripts/build/build_cpp.sh
   ```
   For more details, see the [Cross Compilation](scripts/build/README.md#cross-compilation).

### Running on TI SOC

> **Important:** Before running any examples, ensure that you have properly set up the \<SOC\> environment variables. Missing environment variables are a common cause of errors.

**Model compilation cannot be performed on TI SOC, only inference is supported**.

1. **Transfer compiled model artifacts**:   
   Models must be compiled on an X86 PC first, and then the model artifacts can be transferred to the TI SOC for inference.

   If using NFS mount, the artifacts will already be accessible, else follow the steps below to copy over the compiled artifacts

   ```bash
   scp -r /path/to/edgeai-tidl-tools/runtimes/examples/model-artifacts root@soc:/path/to/edgeai-tidl-tools/runtimes/examples/
   ```

2. **Running Python Examples**:

   ```bash
   cd ./runtimes/examples/python/basic_example/

   # Running onnxrt examples
   python3 basic_example.py --config ./config.yaml -r onnxrt --infer

   # Running tflitert examples
   python3 basic_example.py --config ./config.yaml -r tflitert --infer

   # Running tvmrt examples
   # Note: The TVM python wheel requires additional dependencies that are 
   # currently not provided in SDK 11.2. Please run
   # `pip install psutil typing_extensions` on SoC to enable tvm inference
   # 
   #python3 basic_example.py --config ./config.yaml -r tvmrt --infer

   cd -
   ```

   - Outputs will be saved in `./runtimes/examples/python/basic_example/outputs/{model_name}/offload/frame_{frame_num}/`

2. **Running C++ Examples**:

   ```bash
   cd ./runtimes/examples/cpp/basic_example
   ../bin/Release/basic_example --config ./config.yaml
   cd -
   ```

   - Outputs will be saved in `./runtimes/examples/cpp/basic_example/outputs/{model_name}/offload/frame_{frame_num}/`

## User Guide

This section provides a structured guide to help you navigate through the documentation based on your learning journey with EdgeAI TIDL Tools.

### 1. Getting Started with TIDL

Begin your journey by understanding the compatibility and basic concepts:

- [SDK Version Compatibility](docs/sdk_version_compatibility_table.md): Ensure your SDK version is compatible with this version of EdgeAI TIDL Tools
- [TIDLRT](docs/tidlrt.md): Learn about TIDL's native runtime, its benefits, and when to use it
- [Model Compilation](docs/model_compilation.md): Understand the model compilation process, options, and artifacts
- [Model Inference](docs/model_inference.md): Learn how to run inference with compiled models
- [FAQ](docs/faq.md): Find answers to commonly asked questions to quickly resolve common issues

### 2. Development APIs and Examples

Once you understand the basics, explore how to integrate TIDL into your applications:

- [Python Wrapper APIs](runtimes/tidl_wrapper/python/README.md): Learn how to use the Python APIs for both compilation and inference
- [Python Examples](runtimes/examples/python/README.md): Study reference Python applications to understand implementation patterns
- [C++ Wrapper APIs](runtimes/tidl_wrapper/cpp/README.md): Learn how to use the C++ APIs for inference
- [C++ Examples](runtimes/examples/cpp/README.md): Study reference C++ applications to understand implementation patterns

### 3. Advanced Features

After mastering the basics, explore these topics to leverage advanced features:

- [Supported Operators](docs/operators.md): Check which operators are supported by TIDL
- [Quantization](docs/quantization.md): Learn about quantization methods (PTQ, QAT, mixed precision)
- [Quantization Proto](docs/quantization_proto.md): Advanced quantization parameter configuration
- [Vision Transformers](docs/vision_transformers.md): Learn about vision transformer support and limitations
- [OD Meta Arch](docs/od_meta_arch.md): Understand optimized post-processing for object detection

### 4. Model Optimizations

- [OSRT Model Tools](osrt-model-tools/README.md): Utilities for optimizing ONNX and TFLite models offline

### 5. Performance Optimization

Explore these topics to maximize performance and understand some features in depth:

- [Input/Output Tensors](docs/io_tensors.md): Understand how to handle input and output tensors with TIDL
- [Multi C7x](docs/multi_c7x.md): Learn how to leverage multi-core execution for better performance
- [Preemption](docs/preemption.md): Understand priority scheduling for real-time applications

### 6. Debugging

Finally explore how to troubleshoot.

- [Debugging](docs/debugging.md): Learn how to identify and solve issues with setup, model compilation, inference, and performance


## Test Reports

The repository includes detailed unit test reports for various deep-learning operators across various TI devices in the [docs/reports](docs/reports/README.md) directory. These reports are generated for each release of edgeai-tidl-tools.

These reports are valuable resources for:
- Evaluating which operators and with what properties are supported and tested on specific devices
- Troubleshooting compatibility issues
- Making informed decisions about model selection and optimization

## Additional Support
For any additional queries or support, please visit the [TI E2E Forum](https://e2e.ti.com/).

## License
Please see the license under which this repository is made available: [LICENSE](./LICENSE)
