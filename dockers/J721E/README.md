# J721E Setup
- [J721E Setup](#j721e-setups)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Copying the libraries](#copying-the-libraries)



## Introduction

   - This folder have instructions on how to setup J721E by installing required dependencies.
      

## Setup
- On target run the following. This will download and install required python and system dependencies.
  ```
  cd edgeai-tidl-tools/dockers/J721E
  ./j7_setup.sh
  ```
- The script will display a list of shell variables need to be exported in the end. Copy these vars in your shell.

## Running the examples
- Run the python model compile on PC  [Model Compilation on PC](../../examples/osrt_cpp/README.md#model-compilation-on-pc)
- Run the python examples inside container  [Run Inference on Target](../../examples/osrt_cpp/README.md#model-compilation-on-pc)
- Compile the CPP application by following instruction from [here](../../examples/osrt_cpp/README.md#setup)
