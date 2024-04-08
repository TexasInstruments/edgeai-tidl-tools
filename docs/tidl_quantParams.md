# Quantization Proto : Providing quantization parameters to TIDL

## Introduction

---
TIDL-RT's "quantization proto" mechanism enables you to 
 - Inspect quantization related information for user's model 
 - Allows user to feed quantization control parameters from an external algorithm to modify TIDL's PTQ behavior
 
 "quantization proto" mechanism provides a prototxt based interface to capture information such as "range" and "scale" for each tensor. There are two modes of using this feature
 
 A. **Write Mode**: This information (prototxt file) can be generated during TIDL's compilation stage by specifying a path to prototxt file via the "quant_params_proto_path" field in advanced options. This mode is only active when the file provided via "quant_params_proto_path" doesn't exist.

 B. **Read Mode**: If the file specified by "quant_params_proto_path"  already exists & all fields are correctly populated, the calibration step in TIDL's PTQ get bypassed and scales are derived based on the ranges provided by this file

---

## Steps to provide the range values via quantization proto:

---
  1. Specify the path to your prototxt file as mentioned above via the "quant_params_proto_path"
  2. Only the range values (min, max) fields are used to determine the final scales used by TIDL's quantization module
  3. You can either programmatically generate the prototxt file by following the [specification](#quantization-proto-specification), or use **Write mode** as specified above and then update range values to modify the PTQ behavior
---

## Quantization Proto Specification 

The prototxt file provided is expected to follow the following specification:
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

**In order to be able to bypass the calibration step make sure that:**
- Every layer has minimum and maximum values for all of its tensors
- If any layer has a bias tensor, make sure it has entries of "value" filed equal to "size" field
- TidlTensorQuantParams:min field shall be higher than FLT_MIN and max field shall be lesser than FLT_MAX

Following is an example of a valid prototxt file for a dummy model with 2 layers:
```protobuf text format
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
Reasons for this file being considered valid:
- Number of layers are consistent with number of layer instances
- Each tensor has valid minimum and maximum values
- Bias tensor has "size" = 3 and there are 3 entries of "value" field
  

**Note**: Make sure the file provided by "quant_params_proto_path" doesn't exist.

---

## Steps to troubleshoot
In case you are unable to bypass the calibration step, your file might have one of the following issues:

- Invalid format:
  
    If your file does not have the format per [specification](#quantization-proto-specification), it wont get parsed.

    Following can be an example for invalid format caused by missing texts:
    ```protobuf text format
    ype: SYMMETRIC
    calib_type: PERTENSOR
    layers {
      layer_name: "data"
      layer_type: "TIDL_DataLayer"
      bit_depth: 1
      outpu
    
    layers {
      layer_name: "conv1a"
      layer_type: "TIDL_ConvolutionLayer"
      hts {
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
        valu
      }
    }
    ```

- Inadequate data:
  
  The proto file provided by you might be lacking data for one or more layers that is crucial inorder to bypass calibration.

    Following is an example of a proto file that has inadequate data:
    ```protobuf text format
    num_layers: 2
    quant_type: SYMMETRIC
    calib_type: PERTENSOR
    layers {
      layer_name: "data"
      layer_type: "TIDL_DataLayer"
      bit_depth: 1
      outputs {
        min: FLT_MIN
        max: FLT_MAX
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
        max: FLT_MAX
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
        min: FLT_MIN
        max: 0.53365200757980347
        size: 3
        element_type: 0
        scale: 120.27982330322266
        zero_point: 0
      }
    }
    ```
    What makes it inadequate:
    - Tensors have min/max values set to FLT_MIN/FLT_MAX.
    - Inconsistent Bias values ("size"=3 but number of entries for value is 0)
    - Several typing mistakes such as "ype" instead of "quant_type", "outpu" instead of "output" and many more

---
## Known limitations:
1. The above mechanism is limited to only networks with a single subgraph offloaded to TIDL-RT
2. Only symmetric quantization is currently supported for this mechanism

