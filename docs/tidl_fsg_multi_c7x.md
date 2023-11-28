# Multi-core inference for devices with multiple DSP cores 

## Introduction

Certain SoCs(such as AM69A) have multiple DSP cores coupled with MMA, which can be leveraged to achieve better performance in terms of throughput/latency.
TIDL offering supports using such devices with different inference modes as outlined in the following sections.

## Options to enable multi-core processing
Additional compilation options as explained in [multi core options](../examples/osrt_python/README.md#options-for-devices-with-multiple-DSP-cores) need to be specified in order to utilize this feature. <br>

## Inference modes and target usage

TIDL accepts compilation option "advanced_options:inference_mode", which generates compiled artifacts for the specific inference mode to be used for application.
Supported inference modes are:

### TIDL_inferenceModeDefault (0)
  * This default inference mode enables inference on a single DSP core
  * The specific core to run inference on can be specified using the inference option "core_number" as specified in [multi core options](../examples/osrt_python/README.md#options-for-devices-with-multiple-DSP-cores)
### TIDL_inferenceModeHighThroughput (1)
  * This mode is synonymous to "parallel batch processing" mode
  * It leverages "N" DSP cores to run inference on "N" frames parallelly using "N" instances of the same Deep Neural Network and thus, gives high throughput(high frames per second) for models with multiple batches
  * This inference mode can be coupled with compilation option "advanced_options:num_cores" and inference option "core_start_idx" to process on a subset of DSP cores starting with user defined starting core index
  * Usage:
    * This mode is applicable to cases where same model is intended to process multiple frames in parallel (e.g. multi camera systems)
    * Note that this mode does not give better performance in terms of latency for single frame processing
  * Additional points to note:
    * This mode should not be confused with batch processing on single core as supported in other single DSP devices (AM68PA, AM68A, AM62A, etc.). 
    On single DSP devices, providing a multiple batch model using inference mode TIDL_inferenceModeDefault does internal optimizations to get better performance while still running inference on a single core, as opposed to parallelly inferring batches in TIDL_inferenceModeHighThroughput. This particular optimization is applicable only for very small resolutions and may not be enabled in cases where there is no significant performance benefit. While executing batch inference on multi DSP devices such as AM69A, it is recommended to use TIDL_inferenceModeHighThroughput.
### TIDL_infereneModeLowLatency (2)
  * This mode enables inferring single frame using single instance of a DNN across multiple cores, resulting in lower latency for given frame and a higher abstracted performance in terms of TOPS
  * Similar to TIDL_inferenceModeHighThroughput, this inference mode can be coupled with compilation option "advanced_options:num_cores" and inference option "core_start_idx" to process on a subset of DSP cores starting with user defined starting core index


## Debug Options
Please refer [Troubleshooting](./tidl_osr_debug.md#trace-dump-utility-for-multi-core-inference) for layer level trace dump debug utility for inference mode "TIDL_infereneModeLowLatency".

## Known issues/limitations in current release

1. Inference mode TIDL_inferenceModeHighThroughput is supported only if number of batches is multiple of number of cores used for inference
2. Limitations in TIDL_infereneModeLowLatency: <br>
    * This mode has undergone limited functional validation: <br>
    (1) Following networks have been validated on 2 cores/ 4 cores with inference mode = 2 in this release: Mobilenet v1, Mobilenet v2, Inception net v1, Pointpillars, SSD and Yolov5. <br>
    (2) Following combinations are not supported in this release : (i) Support for high resolution optimization (ii) Inference for tensor bits != 8 (iii) Grouped convolution
    * This mode does not meet desired performance target: Target latency for low latency mode, which is, < 1.2 times (single c7x single DDR instance latency / number of cores) is not met in this release





