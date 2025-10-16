# Build Scripts

This directory contains scripts to help with builds.

## build_cpp.sh

A simple helper script to build C++ runtimes wrapper and examples. This script provides options to build only the runtime wrapper, only the examples, or both. It also supports cross-compilation from x86 hosts to aarch64 targets.

### Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Display help message |
| `--wrapper-only` | Build only the runtime wrapper |
| `--examples-only` | Build only the examples |
| `-j, --jobs N` | Use N parallel jobs for building (default: 2) |
| `-d, --debug` | Build in Debug mode (default: Release) |
| `-c, --clean` | Clean build directories without building |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SOC` | (Required) Target SoC type (e.g., j721s2, am62a) |
| `TIDL_TOOLS_PATH` | (Required) Path to TIDL tools |
| `TARGET_CPU` | (Required for cross-compilation) Target CPU architecture. Set to `aarch64` for cross-compilation|
| `SDK_PATH` | (Required for cross-compilation) Path to the SDK containing `targetfs` and `toolchain` |

### Usage

#### Build both runtime wrapper and examples in Release mode

```bash
export SOC=<your_soc_type>
export TIDL_TOOLS_PATH=<path_to_tidl_tools>
./build_cpp.sh --clean  #Clean the build directories without building
./build_cpp.sh
```

### Notes

- When building examples only, the script checks if the runtime wrapper library is already built. If not, it will build the wrapper first since examples depend on it.
- The script creates build directories if they don't exist.
- Build artifacts are placed in the following locations:
  - Runtime wrapper Libraries: `<REPO_DIR>/runtimes/tidl_wrapper/cpp/lib/<BUILD_TYPE>/`
  - Example Executables:  `<REPO_DIR>/runtimes/examples/cpp/bin/<BUILD_TYPE>/`

### Cross-Compilation

The build system provides option to cross-compile for `aarch64` target from `x86` host cpu.

#### Prerequisites

You need to have SDK present and setup on your x86 cpu since cross-compilation requires
`targetfs` and `toolchain` from the SDK.

> The TARGET_FS_PATH in the build will resolve to `<SDK_PATH>/targetfs`
> The TOOLCHAIN_PATH in the build will resolve to `<SDK_PATH>/toolchain/sysroots/x86_64-arago-linux/usr/bin/aarch64-oe-linux/`

#### Compiling
``` bash
export SOC=<your_soc_type>
export TIDL_TOOLS_PATH=<path_to_tidl_tools>
export SDK_PATH=<path_to_sdk>
export TARGET_CPU=aarch64
./build_cpp.sh --clean  #Clean the build directories without building
./build_cpp.sh
```

#### Configuration File Path in Cross-Compilation

When cross-compiling the basic example, you might encounter issues with the default configuration file path. This is because the `__FILE__` macro used to determine the default config path resolves to the absolute path on the build machine, not the target device.

To address this issue, use the `--config` option when running the example on the target device:

```bash
./basic_example --config ./config.yaml
```

This allows you to specify the correct path to the configuration file on the target device.
