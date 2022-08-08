# Ubuntu 20 Docker Setup
- [Ubuntu 20 Docker Setup](#u20-docker-setups)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Copying the libraries](#copying-the-libraries)



## Introduction

   - This folder have instructions on how to build and deploy Ubuntu 20.04 docker images on Target(Tested on J7ES) 
      

## Setup
- On target run the following. This will create the image and log in to the docker container
- Make sure no existing container is running by ``` docker container ls ```
  ```
  cd edgeai-tidl-tools/dockers/ubuntu_20.04
  ./docker_build.sh
  ./docker_run.sh*
  ```
- Host(In this case J7ES device file system ) volume is mounted at /host inside container
- Run the commands inside container   

  ```
  cd /host/<path_to_edge_ai_tidl_tools>/dockers/ubuntu_20.04
  source container_stup.sh # This will take care of additional dependencies 
  ```

## Running the examples
- Run the python model compile on PC  [Model Compilation on PC](../../examples/osrt_cpp/README.md#model-compilation-on-pc)
- Run the python examples inside container  [Run Inference on Target](../../examples/osrt_cpp/README.md#model-compilation-on-pc)
- Compile the CPP application by following instruction from [here](../../examples/osrt_cpp/README.md#setup)
