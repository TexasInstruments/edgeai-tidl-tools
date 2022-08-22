# X86 Docker Setup
- [Dependency Libs Build](#x86-docker-setups)
  - [Introduction](#introduction)
  - [Copying the Libraries](#copying-the-libraries)



## Introduction

   - This directory is only used to generate the dependencies completely from source. For running the edgeai-tidl-tools the setup script will download and install pre-built dependencies from ti.com. Refer setting setup at
      - [Ubuntu 18.04 docker setup](../ubuntu_18.04/README.md) 
      - [Ubuntu 20.04 docker setup](../ubuntu_20.04/README.md) 
      - [J721E arago linux setup](../J721E/README.md) 
   - OSRT dependencies(libs and python whls) for native arago, Ubuntu 18.04 and Ubuntu 20.04 docker containers are generated using the scripts inside the folders
      - [qemu (dependencies for Ubuntu Docker containers )](qemu/README.md)
      - [x86 (cross compilation of dependencies for native arago linux)](x86/README.md)



## Copying the Libraries
- Once the build (qemu or x86) is completed the generated pywhl/lib can be found at the location mentioned in xxx_build.sh script
- manually copy the required file to your destination 

  