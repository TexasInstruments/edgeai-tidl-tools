# Jupyter Notebooks
- [Jupyter Notebooks](#jupyter-notebooks)
  - [Introduction](#introduction)
  - [EVM IP Address](#evm-ip-address)
  - [Setup](#setup)
  - [Launch Notebook Session](#launch-notebook-session)
  - [Open Jupyter Session in Browser](#open-jupyter-session-in-browser)

## Introduction

Example Jupyter notebooks help to quickly run analytics demos on the C7x-MMA DSP from different open source frameworks. Example notebooks contain OOB demos and custom model notebooks. OOB notebooks demos show how to run inference of a selected model from a list of options. OOB notebooks showcase main framework APIs for running inference, gives an example of pre and post-processing functions and report back performance metrics.

On the other hand, custom models, take a pre-trained classification model and compile it using one of the framework compilers to generate deployable artifacts, which can be run on the target by using frameworks inference APIs.

Analytics OOB demos and frameworks covered:
- Image Classification TFLite
- Image Classification ONNX
- Image Classification DLR
- Image Detection TFLITE
- Image Detection DLR
- Image Detection ONNX
- Image Segmentation TFlite
- Image Segmentation ONNX
- Image Segmentation DLR

Below Custom Model compialtion and infrnce is supported only on PC emualtion:
- TFLite Custom Model
- ONNXRT Custom Model
- TVM/NEO-AI-DLR Custom Model

## EVM IP Address
- Connect to your EVM
- Note your EVM IP address
    ```
    ifconfig
    ```
## Setup
- Prepare the Environment for the Model compilation by follwoing the setup section [here](../../
README.md#setup)

## Launch Notebook Session
- If this is the first time run below script to get your setup ready and copy "token" from printed information in the EVM. If not the first time, go to step 4
    ```
    source ./launch_notebook.sh
    ```
- token example: http://j7-evm:8888/?token=5f4dc2f9c60318a8d8f70013a0786d56fd1c17a012a6a630 
- If this is not the first time, It is only needed to setup environment variables and relaunch Jupyter notebook server. No need to re-run notebook_evm.sh
  ```
  source ./launch_notebook.sh --skip_setup --skip_models_download 
  ```

## Open Jupyter Session in Browser
- In a webbrowser open Jupiter notebook using EVM IP address and token
    - ex: http://192.168.1.199:8888/tree
    - if asked paste token
