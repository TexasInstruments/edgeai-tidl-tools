# Quantization Proto: Providing Quantization Parameters to TIDL

## Table of Contents

- [Introduction](#introduction)
- [Providing Range Values via Quantization Proto](#providing-range-values-via-quantization-proto)
- [Quantization Proto Specification](#quantization-proto-specification)
- [Requirements for Bypassing Calibration](#requirements-for-bypassing-calibration)
- [Example](#example)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)

## Introduction

The TIDL's "quantization proto" mechanism is a powerful feature that enables users to:

- Inspect quantization-related information for the models
- Feed quantization control parameters from external algorithms to modify TIDL's Post-Training Quantization (PTQ) behavior

This mechanism provides a prototxt-based interface to capture critical information such as "range" and "scale" for each tensor in the model. This feature can be used in two modes:

1. **Write Mode** - Generate a prototxt file during TIDL's compilation stage by specifying a path via the `advanced_options:quant_params_proto_path` compilation option. This mode is active only when the file specified doesn't already exist.

2. **Read Mode** - If the file specified by `advanced_options:quant_params_proto_path` already exists and is correctly populated, TIDL bypasses the calibration step in its PTQ process. Instead, it derives scales based on the ranges provided in your file.

## Providing Range Values via Quantization Proto

To leverage this feature effectively:

1. Specify the path to your prototxt file via the `advanced_options:quant_params_proto_path` parameter
2. Note that only the range values (min, max) are used to determine the final scales used by TIDL's quantization module
3. You can either:
   - Programmatically generate the prototxt file following the specification below in *Quantization Proto Specification* section
   - Use **Write mode** to generate an initial file and then update the range values to modify PTQ behavior

## Quantization Proto Specification

The prototxt file must follow this specification:

```protobuf
// Top level configuration for the message storing TIDL Network Quantization Parameters
message TIDLNetQuantParams {
  optional int32 num_layers = 1;
  optional TidlQuantType quant_type = 2 [default = SYMMETRIC];
  optional TidlCalibType calib_type = 3 [default = PERTENSOR];
  repeated TidlLayerQuantParams layers = 4;
}

// Quantization Type
enum TidlQuantType {
SYMMETRIC = 1;
ASYMMETRIC = 2;
}

// Calibration Type
enum TidlCalibType {
PERTENSOR = 1;
PERCHANNEL = 2;
}

// Configuration for storing TIDL Layer Quantization Parameters
message TidlLayerQuantParams {
  optional string layer_name = 1;
  optional string layer_type = 2;
  optional uint32 bit_depth = 3 [default = 8];
  repeated TidlTensorQuantParams outputs = 4;
  repeated TidlTensorQuantParams weights = 5;
  repeated TidlTensorQuantParams bias = 6;
  repeated TidlTensorQuantParams slope = 7;
}

// Configuration for storing TIDL Tensor Quantization Parameters
message TidlTensorQuantParams {
  optional float min = 1 [default = -3.4028234664e+38];
  optional float max = 2 [default = 3.4028234664e+38];
  optional uint32 size = 3;
  optional uint32 element_type = 4;
  repeated double scale = 5 [packed=true];
  repeated uint32 zero_point = 6 [packed=true];
  repeated double value = 7 [packed=true];
}
```

## Requirements for Bypassing Calibration

To successfully bypass the calibration step, ensure that:

- Every layer has minimum and maximum values specified for all of its tensors
- If any layer has a bias tensor, make sure it has entries of "value" field equal to the "size" field
- The `min` field should be higher than `FLT_MIN` and the `max` field should be lesser than `FLT_MAX`

## Example

Here's an example of a valid prototxt file for a model with 2 layers:

```protobuf
num_layers: 2
quant_type: SYMMETRIC
calib_type: PERTENSOR
layers {
  layer_name: "data"
  layer_type: "TIDL_DataLayer"
  bit_depth: 1
  outputs {
    min: 0
    max: 255
    size: 1
    element_type: 0
    scale: 1
    zero_point: 0
  }
}
layers {
  layer_name: "conv1a"
  layer_type: "TIDL_ConvolutionLayer"
  bit_depth: 1
  outputs {
    min: 0
    max: 4.09663105
    size: 1
    element_type: 0
    scale: 56.387794494628906
    zero_point: 0
  }
  weights {
    min: -0.00830631796
    max: 0.00886716694
    size: 1
    element_type: 0
    scale: 14435.275390625
    zero_point: 0
  }
  bias {
    min: -0.52766174077987671
    max: 0.53365200757980347
    size: 3
    element_type: 0
    scale: 120.27982330322266
    zero_point: 0
    value: 0.28929620981216431
    value: -0.52766174077987671
    value: 0.53365200757980347
  }
}
```

### What Makes This Example Valid:
- The number of layers is consistent with the number of layer instances
- Each tensor has valid minimum and maximum values
- The bias tensor has "size" = 3 and there are 3 entries of "value" field

## Troubleshooting

If you're unable to bypass the calibration step, your file might have one of these issues:

### 1. Invalid Format

If your file doesn't follow the specification defined by *Quantization Proto Specification* section above, the file will not be parsed correctly. These issues include:
- Missing fields
- Incomplete entries
- Syntax errors

### 2. Inadequate Data

Your proto file might lack data crucial for bypassing calibration:
- Tensors with min/max values set to `FLT_MIN`/`FLT_MAX`
- Inconsistent bias values (e.g., "size"=3 but number of entries for value is 0)
- Missing required fields for certain layers or tensors

## Known Limitations

1. This mechanism is limited to networks with a single subgraph fully offloaded to TIDL
2. Currently, only symmetric quantization is supported for this mechanism
