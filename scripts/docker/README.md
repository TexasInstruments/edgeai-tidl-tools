# Docker Setup for EdgeAI TIDL Tools

This directory contains scripts and configuration files for setting up and running EdgeAI TIDL Tools in Docker containers. Docker provides an isolated environment with all the necessary dependencies pre-installed, making it easier to use the tools across different systems.

## Overview

The Docker setup supports both CPU and GPU-based TIDL tools:

- **CPU Tools**: Default configuration using Ubuntu 22.04 base image
- **GPU Tools**: NVIDIA GPU-accelerated configuration using NVIDIA HPC SDK with CUDA support

## Setup Instructions

The following environment variables can be exported to configure the docker:
<div align="center">

| Environment Variable | Default | Available Options | Notes |
|:---------------------|:--------|:------------------|:------|
| TIDL_TOOLS_TYPE | CPU | CPU or GPU| Setting TIDL_TOOLS_TYPE=GPU sets up the docker image for tidl tools built with OpenACC based acceleration|
| REPO_LOCATION | Unset | ANY | Repo location for pulling docker image in case of a private network|
| PROXY | Unset | ANY | PROXY to be used in case of a private network|
</div>

> Note: In case you are under TI's network, set REPO_LOCATION=artifactory.itg.ti.com/docker-public/library/ and PROXY=*TI WEBPROXY*

### Docker image for CPU tools

#### 1. Change CWD to repo root directory

```bash
cd <EDGEAI_TIDL_TOOLS_REPO_DIR>
```

#### 2. Install Docker and Required Dependencies

```bash
TIDL_TOOLS_TYPE=CPU ./scripts/docker/setup_docker.sh
```

This script will:
- Install Docker if not already installed
- Add your user to the Docker group to run Docker without sudo

#### 3. Build the Docker Image

```bash
TIDL_TOOLS_TYPE=CPU ./scripts/docker/build_docker.sh
```
The script will:
- Build ubuntu22.04 docker image from `Dockerfile` with name `edgeai_tidl_tools_x86_ubuntu_22`

#### 4. Run the Docker Container

```bash
TIDL_TOOLS_TYPE=CPU ./scripts/docker/run_docker.sh
```

The script will:
- Mount the project root directory to `/home/root` in the container
- Set up shared memory (4GB)
- Start an interactive terminal session


### Docker image for GPU tools

#### 1. Change CWD to repo root directory

```bash
cd <EDGEAI_TIDL_TOOLS_REPO_DIR>
```

#### 2. Install Docker and Required Dependencies

```bash
TIDL_TOOLS_TYPE=GPU ./scripts/docker/setup_docker.sh
```

This script will:
- Install Docker if not already installed
- Add your user to the Docker group to run Docker without sudo
- Install NVIDIA Container Toolkit

#### 2. Build the Docker Image

```bash
TIDL_TOOLS_TYPE=GPU ./scripts/docker/build_docker.sh
```
The script will:
- Build nvhpc based ubuntu22.04 docker image from `Dockerfile_GPU` with name `edgeai_tidl_tools_x86_ubuntu_22_gpu`

#### 3. Run the Docker Container

```bash
TIDL_TOOLS_TYPE=GPU ./scripts/docker/run_docker.sh
```

The script will:
- Mount the project root directory to `/home/root` in the container
- Set up shared memory (4GB)
- Start an interactive terminal session
- Enable GPU access if using GPU tools
- In case Docker is not able to detect GPU during run_docker, please install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Notes

- The container mounts the project root directory, so any changes made inside the container to files in this directory will persist outside the container.
- For GPU support, ensure you have the NVIDIA Container Toolkit installed and configured properly.
- The Docker images include all dependencies required for EdgeAI TIDL Tools development and usage.
