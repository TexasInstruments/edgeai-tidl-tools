
<!-- TOC -->

- [Advanced Setup](#advanced-setup)
  - [Docker based setup for X86\_PC](#docker-based-setup-for-x86_pc)
  - [Advanced Setup Options](#advanced-setup-options)

<!-- /TOC -->

# Advanced Setup

## Docker based setup for X86_PC
The following environment variables must be exported to configure the docker based setup (export ENV_VAR:Option):
<div align="center">

| No | Environment Variable | Default | Available Options | Notes |
|:--:|:---------------------|:--------|:------------------|:------|
| 1  | SOC | Unset | AM62A, AM67A, AM68A, AM69A, AM68PA | Must be set to the appropriate device|
| 2  | TIDL_TOOLS_TYPE | Unset | Unset, GPU| Setting TIDL_TOOLS_TYPE=GPU sets up the docker image for tidl tools built with OpenACC based acceleration<br /> GPU tools require a NVIDIA GPU with CUDA Support [^1]|
| 3  | USE_PROXY | Unset | Unset, ti | Should only be set if within TI network|
</div>


Steps to build and run the docker image:

1. Clone Github repo

          git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
          cd edgeai-tidl-tools
          git checkout <TAG Compatible with your SDK version>
2. One time setup for Docker:

          source ./scripts/docker/setup_docker.sh

3. Build the docker image:
          
          source ./scripts/docker/build_docker.sh

4. Run the docker image:

          source ./scripts/docker/run_docker.sh

5. After running the docker container, run the setup script:

        cd /home/root/
        # Supported SOC name strings am62, am62a, am67a, am68a, am68pa, am69a
        export SOC=<Your SOC name>
        #Set TIDL_TOOLS_TYPE to GPU if using GPU tools
        export TIDL_TOOLS_TYPE=<Your Tools Type (GPU)>
        source ./setup.sh


## Troubleshooting GPU Docker Setup Issues:
- In case Docker is not able to detect your GPU during the run_docker step, you need to install the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)<br>
- After installing the above, restart Docker: 

        sudo systemctl restart docker
- Above steps have been validated on a 2080 TI GPU with CUDA Version 12.2 & Driver versions 535.86 <br>

## Advanced Setup Options
  - To validate only  python examples and avoid running CPP examples, invoke the setup script with below option

```
 source ./setup.sh --skip_cpp_deps
```
  - To use existing ARM GCC tools chain installed  and avoid downloading, invoke the setup script with below option and set "ARM64_GCC_PATH" environment variable

```
export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
source ./setup.sh --skip_arm_gcc_download
```

  - If you are building the PSDK-RTOS/FIRMWARE-BUILDER from source and updating any of the TIDL tools during the development, then set  "TIDL_TOOLS_PATH" environment variable before starting setup script

```
export TIDL_TOOLS_PATH=$PSDKR_INSTALL_PATH/tidl_xxx_xx_xx_xx_xx/tidl_tools
source ./setup.sh
```

