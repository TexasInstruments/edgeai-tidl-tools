# Jupyter Notebooks
- [Jupyter Notebooks](#jupyter-notebooks)
  - [Introduction](#introduction)
  - [EVM IP Address](#evm-ip-address)
  - [Setup](#setup)
  - [Launch Notebook Session](#launch-notebook-session)
  - [Open Jupyter Session in Browser](#open-jupyter-session-in-browser)

## Introduction

Example Jupyter notebooks help to quickly run neural networks on the C7x-MMA DSP from different open source frameworks. Example notebooks contain OOB demos and custom model notebooks. OOB notebooks show how to run inference of a selected model from a list of options. OOB notebooks showcase main framework APIs for running inference, gives an example of pre and post-processing functions and report back performance metrics.

On the other hand, custom models, take a pre-trained classification model and compile it using one of the framework compilers to generate deployable artifacts, which can be run on the target by using frameworks inference APIs.

Neural network OOB demos and frameworks covered:
- Image Classification TFLite
- Image Classification ONNX
- Image Classification DLR
- Image Detection TFLITE
- Image Detection DLR
- Image Detection ONNX
- Image Segmentation TFlite
- Image Segmentation ONNX
- Image Segmentation DLR

Below Custom Model compilation and inference is supported only on host emulation:
- TFLite Custom Model
- ONNXRT Custom Model
- TVM/NEO-AI-DLR Custom Model

## Note your IP Address - To use after "Launch Notebook Session step"
### If running on EVM
- Connect to your EVM
- Note your EVM IP address
    ```
    ifconfig
    ```
### If running on Host Emulation
- Use your host IP address to access the notebooks
### If running on Host Emulation inside a docker
- Run the docker container with -p option
    Example: sudo docker run -it -p 8888:8888 your_docker_image
- Use your host IP address to access the notebooks

## Setup
- Prepare the Environment for the Model compilation by following the setup section [here](../../README.md#setup)

## Launch Notebook Session
- If this is the first time, run below script to get your setup ready and copy "token" from printed information in the EVM. If not the first time, go to "Open Jupyter Session in Browser" section
    ```
    source ./launch_notebook.sh
    ```
- token example: http://j7-evm:8888/?token=5f4dc2f9c60318a8d8f70013a0786d56fd1c17a012a6a630 
- If this is not the first time, It is only needed to setup environment variables and relaunch Jupyter notebook server.
  ```
  source ./launch_notebook.sh --skip_setup --skip_models_download 
  ```

## Open Jupyter Session in Browser
- In a web browser open Jupyter notebook using EVM's or Host IP address and token
    - ex: http://192.168.1.199:8888/tree
    - if asked paste token
