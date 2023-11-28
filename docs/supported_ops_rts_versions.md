# Supported Operators & Runtimes

## Operator Mapping (TIDL-RT):

<div align="center">

| No | TIDL Layer Type                | ONNX Ops                                    | TFLite Ops                                 | Notes |
|:--:|:-------------------------------|:--------------------------------------------|:-------------------------------------------|:------|
| 1  | TIDL_ConvolutionLayer          | Conv                                        | CONV_2D<br>DEPTHWISE_CONV_2D               | <ul><li>Regular & Depthwise convolution will be imported as convolution </li><li> For TFLite DepthwiseConv2dNative, depth_multiplier shall be 1 if number of input channels > 1 </li><li>ReLU & Batchnorm layers will be merged into convolution to get better performance</li><li>Validated kernel sizes: 1x1, 3x3, 5x5, 7x7,1x3,3x1,1x5,5x1,1x7,7x1</li><li> If stride == 4, only supported kernel == 11x11</li><li>if stride == 2, kernel should be less than 7. Even kernel dimensions like 2x2, 4x4, 6x6 are not supported</li><li>Depthwise Separable Convolution only supports 3x3,5x5,7x7 with stride 1 and 3x3 with stride 2</li><li> Convolutions with kernel NxN, stride = N (**Patch embedding**) are also supported but are not currently optimal in this release </li><li> Dilated Convolution is only supported for non-strided convolution</li></ul> **Note : Please refer to MMALIB's release notes in your SDK for all supported configurations**<br> **Note : Some of the kernel combinations are not optimized in the current release, please refer to MMALIB's release notes for the same** |
| 2  | TIDL_BatchNormLayer            | BatchNormalization                          |                                            | <ul><li>ReLU, Scale, Bias, PReLU, Leaky ReLU, Hard Sigmoid, TanH, ELU & GELU will be merged & imported as batchnorm</li><li> All channel-wise broadcast operations are mapped to Batchnorm</li></ul>|
| 3  | TIDL_PoolingLayer              | MaxPool<br>AveragePool<br>GlobalAveragePool | MAX_POOL_2D<br>AVERAGE_POOL_2D<br>MEAN     | <ul><li>Pooling has been validated for the following kernel sizes: 3x3,2x2,1x1, with a maximum stride of 2</li></ul> |
| 4  | TIDL_EltWiseLayer              | Add<br>Mul                                  | ADD<br>MUL                                 | <ul><li>Support for 2 tensors validated extensively, multiple input tensors have had limited validation</li></ul> |
| 5  | TIDL_InnerProductLayer         | Gemm, MatMul                                | FULLY_CONNECTED                            | <ul><li>Supports up to 3 dimension tensors only. Any higher dimension tensor shall have dimension value equal to 1 for fourth dimension onwards </li><li> Doesn’t support broadcast of 3rd dimension for variable inputs  </li><li>For TDA4VM variable input case, doesn’t support unsigned input </li><li> Support for MatMul with both variable tensors is not optimal in the current release</li><li>Higher dimensional matmuls can be realized by reshaping the higher dimensions into the 3rd dimension</li></ul>|
| 6  | TIDL_SoftMaxLayer              | Softmax                                     | SOFTMAX                                    | <ul><li>Supports 8-bit inputs with 8-bit outputs with axis support for width (axis=-1) for any NxCxHxW tensor</li><li>Supports integer (8/16-bit) to float softmax only for flattened inputs</li><li>Softmax performance is not optimal in the current release</li></ul> |
| 7  | TIDL_Deconv2DLayer             | ConvTranspose                               | TRANSPOSE_CONV                             | <ul><li>Only 8x8, 4x4 and 2x2 kernel with 2x2 stride is supported. It is recommended to use Resize/Upsample to get better performance. This layer is not supported in 16-bit for AM62A</li></ul>|
| 8  | TIDL_ConcatLayer               | Concat                                      | CONCATENATION                              | <ul><li>Concat defaults channel-wise by default. Concat will be width-wise if it happens post a flatten layer (used in the context of SSD)</li></ul>|
| 9  | TIDL_SliceLayer                | Split                                       | NA                                         | <ul><li>Only channel wise slice is supported (with no stride)</li><li>Patch merging expressed with strided slice will be transformed into a transpose layer</li></ul>|
| 10 | TIDL_CropLayer                 | NA                                          | NA                                         |  |
| 11 | TIDL_FlattenLayer              | Flatten                                     | NA                                         | <ul><li>16-bit is not optimal in the current version</li></ul>|
| 12 | TIDL_ArgMaxLayer               | ArgMax                                      | ARG_MAX                                    | <ul><li>Only axis == 1 is supported (For Semantic Segmentation) </li></ul>|
| 13 | TIDL_DetectionOutputLayer      | NA                                          | NA                                         | <ul><li>Please refer to the [Meta Architecture Documentation](./tidl_fsg_od_meta_arch.md) for further details </li></ul>|
| 14 | TIDL_ShuffleChannelLayer       | Reshape + Transpose + Reshape               | NA                                         |  |
| 15 | TIDL_ResizeLayer               | UpSample                                    | RESIZE_NEAREST_NEIGHBOR<br>RESIZE_BILINEAR | <ul><li>Only power of 2 and symmetric resize is supported <br>Any resize ratio which is power of 2 and greater than 4 will be placed by combination of 4x4 resize layer and 2x2 resize layer <br> For example, an 8x8 resize will be replaced by a 4x4 resize followed by a 2x2 resize </li></ul> |
| 16 | TIDL_DepthToSpaceLayer         | DepthToSpace                                | DEPTH_TO_SPACE                             |  <ul><li>Supports non-strided convolution with upscale factors of 2, 4 and 8. This layer is currently not supported for AM62A </li></ul>| 
| 17 | TIDL_SigmoidLayer              | Sigmoid/Logistic                            | SIGMOID/LOGISTIC                           |   |
| 18 | TIDL_PadLayer                  | Pad                                         | PAD                                        |   |
| 19 | TIDL_ColorConversionLayer      | NA                                          | NA                                         |  <ul><li>Only YUV420 NV12 format conversion to RGB/BGR color format is supported </li></ul>|
| 20 | TIDL_BatchReshapeLayer         | NA                                          | NA                                         |  |
| 21 | TIDL_DataConvertLayer          | NA                                          | NA                                         |  |
| 22 | TIDL_ReshapeLayer              | Reshape                                     | RESHAPE                                         |  |
| 23 | TIDL_ScatterElementsLayer                 | ScatterND | NA | <ul><li>It is supported with following constraints:<ul><li> 'data' input is ignored and assumed as complete zero value buffer </li><li> 'indices' with only int32 data type, even though operator interface is int64</li><li> 'updates' data type can be int8,int16,uint8,uint16</li><li> 'element' and 'line/vector' updates are only supported for now</li><li> The datatype of the ‘update’ is same as the data type of the input ‘data’</li><li> ‘index_put_’ operator in pytorch can be used to generate this operator in the onnx mode</li></ul></li></ul>
| 24 | TIDL_GatherLayer               | Gather                                      | NA                                         | <ul><li>It is supported with following constraints:<ul><li> 'line/vector' gathers are only supported for now</li><li>  ‘index_select’ operator in pytorch can be used to generate this operator in the onnx mode with restriction on indices tensor to be one-dimensional </li></ul></li></ul>|
| 25 | TIDL_TransposeLayer            | Transpose                                   | TRANSPOSE                                  | <ul><li>Support for 4D Transpose is enabled, i.e every possible permutation of (Batch, Channel, Height, Width) is supported </li></ul>|
| 26 | TIDL_LayernormLayer            | ReduceMean-Sub-Pow(2)-ReduceMean-Add-Sqrt-Div                                   | NA                                  | <ul><li>Only 8-bit signed input is supported</li><li>Only supports the width axis (axis=-1)</li><li>Performance in current release is suboptimal</li></ul>|

</div>
<br>

## Other compatible layers

<div align="center">

| No | ONNX Ops  | TFLite Ops    | Notes |
|:--:|:----------|:--------------|-------|
| 1  | Split     |               | Split layer will be removed after import |
| 2  |            | MINIMUM       | For ReLU6 / ReLU8      |
| 3 | Clip      |               | Parametric activation threshold PACT       |

</div>
<br>


# Supported model formats & operator versions
Proto files from the versions below are used for validating pre-trained models. In most cases, models from new versions should also work since the core operators tend to remain the same
  - ONNX - 1.3.0 (opset 18)
  - TFLite - Tensorflow 2.0-Alpha

*Since the Tensorflow 2.0 is planning to drop support for frozen buffer, we recommend to users to migrate to TFLite model format for Tensorflow 1.x.x as well. TFLite model format is supported in both TF 1.x.x and TF 2.x*


# Feature set comparison across devices

<div align="center">

| Feature  | AM62A | AM68A |AM68PA | AM69A|
| ------- |:-----------:|:-----------:|:-----------:|:-----------:|
|Support for Asymmetric, Per Channel Quantization <br> ([Asymmetric, per-axis quantization](tidl_fsg_quantization.md#d-native-support-for-tensorflow-lite-int8-ptq-models))  | :heavy_check_mark: |:heavy_check_mark: | :x: |:heavy_check_mark:|
| Support for LUT accelerated non-linear activations<sup>1</sup>  | :x: |:heavy_check_mark: | :heavy_check_mark:|:heavy_check_mark:|

</div>

*<sup>1</sup>LUT accelerated non-linear activations include  Sigmoid, Hard Sigmoid, GELU, TanH, Softmax & ELU*