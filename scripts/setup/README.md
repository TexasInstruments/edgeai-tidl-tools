# TIDL Tools Setup Scripts

This directory contains scripts for setting up and configuring the TIDL (Texas Instruments Deep Learning) tools environment. These scripts help with downloading, installing, and configuring the necessary components for using TIDL tools with various TI SoCs (System on Chips).

## Table of Contents
- [X86 Setup Scripts Overview](#x86-setup-scripts-overview)
  - [setup.sh](#setupsh)
  - [setup_env.sh](#setup_envsh)
  - [Workflow](#workflow)
- [TARGET Setup Scripts Overview](#target-setup-scripts-overview)
  - [update_target.sh](#update_targetsh)

## X86 Setup Scripts Overview

### setup.sh

`setup.sh` is the main setup script that downloads and installs TIDL tools for one or more SoCs.

#### Key Features:
- Downloads TIDL tools for all supported SoCs under **repo_dir/tools**
- Downloads Python dependencies and OSRT wheels
- Downloads C++ dependency for OSRT
- Downloads out-of-box data (models and inputs) for examples
- Includes comprehensive error handling for robust execution
- Can be run from any directory, as it correctly references its own location

#### Usage:
```bash
# To download TIDL tools for all supported SOCs
./setup.sh

# To download TIDL tools for a specific SOC
export SOC=J721S2
./setup.sh

# Additional options
./setup.sh --skip_model_optimizer     # Skip installing model optimizer python package
./setup.sh --skip_cpp_deps            # Skip downloading dependencies for CPP examples
./setup.sh --skip_data                # Skip downloading out-of-box data
```

#### Supported SOCs:
  - AM62
  - AM62A
  - J721E  | TDA4VM
  - J721S2 | TDA4VL  | AM68A
  - J784S4 | TDA4VH  | AM69A
  - J722S  | TDA4AEN | AM67A

### setup_env.sh

`setup_env.sh` sets up the environment variables required for using TIDL tools with a specific SOC.

#### Key Features:
- Sets up environment variables like SOC, TIDL_TOOLS_PATH, LD_LIBRARY_PATH, etc.
- Verifies that the TIDL tools for the specified SOC are installed
- Can be sourced from any directory, as it correctly references its own location

#### Usage:
```bash
# To set up the environment for a specific SOC
source setup_env.sh J721S2
```

### Workflow

A typical workflow for using these scripts on x86 machine:

1. Run `setup.sh` on your development machine to download TIDL tools for your target SoC(s). The tools will be downloaded under repo_dir/tools/
2. Source `setup_env.sh` to set up the environment for a specific SoC
3. Now you have the environment ready for TIDL model compilation and inference on x86 machine


## TARGET Setup Scripts Overview

### update_target.sh

`update_target.sh` is used to update a target device with firmware, components, and libraries from a **newer TIDL tools version while using a previous SDK version** (enabling backward compatibility). The tools version will clearly call out that if this is possible in the [SDK Version Compatibility](../../docs/sdk_version_compatibility_table.md).

> **_NOTE:_**
> - This script is meant to be invoked on the **target device only**.
> - Make sure you have a **stable internet connection** on the target device to run this script.
> - Make sure you **reboot** the device after the update for the new firmware to be loaded.
> - If you are running these scripts from within the **TI network**, you need to set proxy environment variables before running the scripts

#### Key Features:
- Updates TIDL headers
- Updates OSRT (Open Standard Runtime) components on the target device
- Updates pre-built firmware and libraries on the target device
- Provides control over what components to update through environment variables

#### Components Updated on Target Device Filesystem:

 - arm-tidl headers 
 - C7X firmwares and TIDL arm side libraries. This can be controlled by `UPDATE_FIRMWARE_AND_LIB` flag
 - onnxruntime, tflite_runtime, tvm, and tidlruntime wheels, headers and libraries. This can be controlled by `UPDATE_OSRT_COMPONENTS` flag
 - Build and install cnpy if not present

#### Environment Variables

The `update_target.sh` script uses several environment variables to control its behavior:

- `SOC`: Specifies the target System on Chip.
- `TISDK_IMAGE`: Specifies the type of TISDK image on the target device
- `SDK_VERSION`: Specifies the SDK version on the target device
- `UPDATE_OSRT_COMPONENTS`: Controls whether to update OSRT components
- `UPDATE_FIRMWARE_AND_LIB`: Controls whether to update firmware and libraries

#### <u>SDK 11.0 and 11.1</u>

Run the following on target device to update components:
```bash
export SDK_VERSION=11_1             # Choose from: 11_0, 11_1
export SOC=J721S2                   
export TISDK_IMAGE=adas             # Choose: adas (for EVM boards), edgeai (for SK boards)
./update_target.sh
```

> **_NOTE:_**
> - SDK_VERSION 11_0 does not exist for AM62A
> - AM62A does not have ADAS Image. Use EDGEAI instead

You can also control which components to update:
```bash
# Update only OSRT components, skip firmware and libraries
export UPDATE_OSRT_COMPONENTS=1
export UPDATE_FIRMWARE_AND_LIB=0
./update_target.sh

# Update only firmware and libraries, skip OSRT components
export UPDATE_OSRT_COMPONENTS=0
export UPDATE_FIRMWARE_AND_LIB=1
./update_target.sh
```

> **_NOTE:_**
> Make sure you **reboot** the device after the update for the new firmware to be loaded.
