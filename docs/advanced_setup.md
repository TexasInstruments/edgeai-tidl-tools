
<!-- TOC -->

- [Advanced Setup](#advanced-setup)
  - [Docker based setup for X86\_PC](#docker-based-setup-for-x86_pc)
  - [Advanced Setup Options](#advanced-setup-options)

<!-- /TOC -->

# Advanced Setup

## Docker based setup for X86_PC

Detailed steps to use Docker based setup for X86_PC
1. Install Docker if it is not already installed
   
          sudo apt-get install docker.io

   
1. Clone Github repo
          
          git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
          cd edgeai-tidl-tools
          git checkout <TAG Compatible with your SDK version>

1. Steps to add docker to sudoers group

          sudo groupadd docker
          sudo usermod -aG docker $USER
          newgrp docker # to reflect the changes in current session 

1. Build Docker Image
          
          sudo docker build -f Dockerfile -t x86_ubuntu_18 .
          # To build docker image in TI's internal network,  run below instead
          sudo docker build --build-arg REPO_LOCATION=artifactory.itg.ti.com/docker-public/library/ --build-arg USE_PROXY=ti  -f Dockerfile -t x86_ubuntu_18 .


1. Run the Docker
          
          sudo docker run -it --shm-size=4096m --mount source=$(pwd),target=/home/root,type=bind x86_ubuntu_18

2. When run above you would get root prompt and edgeai-tidl-tools is mounted at /home/root. Now, run below to complete the setup 

        cd /home/root/
        # Supported SOC name strings am62, am62a, am68a, am68pa, am69a
        export SOC=<Your SOC name>
        source ./setup.sh 


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

