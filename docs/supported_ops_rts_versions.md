# Supported Operators & Runtimes

## Operator Mapping (TIDL-RT):

<div align="center">

| No | TIDL Layer Type                | ONNX Ops                                    | TFLite Ops                                 | Notes |
|:--:|:-------------------------------|:--------------------------------------------|:-------------------------------------------|:------|
| 1  | TIDL_ConvolutionLayer          | Conv                                        | CONV_2D<br>DEPTHWISE_CONV_2D               | Regular & Depthwise convolution will be imported as convolution <br> For TFLite DepthwiseConv2dNative, depth_multiplier shall be 1 if number of input channels > 1. <br> ReLU & Batchnorm layers will be merged into convolution to get better performance<br>Validated kernel sizes: 1x1, 3x3, 5x5, 7x7,1x3,3x1,1x5,5x1,1x7,7x1.<br> If stride == 4, only supported kernel == 11x11.<br>if stride == 2, kernel should be less than 7. Even kernel dimensions like 2x2, 4x4, 6x6 are not supported.<br>Depthwise Separable Convolution only supports 3x3,5x5,7x7 with stride 1 and 3x3 with stride 2.<br> Dilated Convolution is only supported for non-strided convolution<br> **Note : Please refer to MMALIB's release notes in your SDK for all supported configuration**<br> **Note : Some of the kernel combinations are not optimized in the current release, please refer to MMALIB's release notes for the same** |
| 2  | TIDL_BatchNormLayer            | BatchNormalization                          |                                            | ReLU, Scale, Bias, PReLU, Leaky ReLU, Hard Sigmoid & ELU will be merged & imported as batchnorm<br> All channel-wise broadcast operations are mapped to batchnorm |
| 3  | TIDL_PoolingLayer              | MaxPool<br>AveragePool<br>GlobalAveragePool | MAX_POOL_2D<br>AVERAGE_POOL_2D<br>MEAN     | Pooling has been validated for the following kernel sizes: 3x3,2x2,1x1, with a maximum stride of 2 |
| 4  | TIDL_EltWiseLayer              | Add<br>Mul                                  | ADD<br>MUL                                 | Support for 2 tensors validated extensively, multiple input tensors have had limited validation |
| 5  | TIDL_InnerProductLayer         | Gemm                                        | FULLY_CONNECTED                            | Input shape must be 1x1x1xN.Please use global pooling/flatten before innerproduct<br>Feature size larger than 2048*2048 is not optimal |
| 6  | TIDL_SoftMaxLayer              | Softmax                                     | SOFTMAX                                    | Input shape must be 1x1x1xN. Please use global pooling/flatten before softmax. |
| 7  | TIDL_Deconv2DLayer             | ConvTranspose                               | TRANSPOSE_CONV                             | Only 8x8, 4x4 and 2x2 kernel with 2x2 stride is supported. It is recommended to use Resize/Upsample to get better performance|
| 8  | TIDL_ConcatLayer               | Concat                                      | CONCATENATION                              | Concat defaults channel-wise by default. Concat will be width-wise if it happens post a flatten layer (used in the context of SSD)|
| 9  | TIDL_SliceLayer                | Split                                       | NA                                         | Only channel wise slice is supported |
| 10 | TIDL_CropLayer                 | NA                                          | NA                                         |  |
| 11 | TIDL_FlattenLayer              | Flatten                                     | NA                                         | 16-bit is not optimal in the current version|
| 12 | TIDL_ArgMaxLayer               | ArgMax                                      | ARG_MAX                                    | Only axis == 1 is supported (For Semantic Segmentation) |
| 13 | TIDL_DetectionOutputLayer      | NA                                          | NA                                         | Please refer to the [Meta Architecture Documentation](./tidl_fsg_od_meta_arch.md) for further details |
| 14 | TIDL_ShuffleChannelLayer       | Reshape + Transpose + Reshape               | NA                                         |  |
| 15 | TIDL_ResizeLayer               | UpSample                                    | RESIZE_NEAREST_NEIGHBOR<br>RESIZE_BILINEAR | Only power of 2 and symmetric resize is supported <br>Any resize ratio which is power of 2 and greater than 4 will be placed by combination of 4x4 resize layer and 2x2 resize layer <br> For example, an 8x8 resize will be replaced by a 4x4 resize followed by a 2x2 resize  |
| 16 | TIDL_DepthToSpaceLayer         | DepthToSpace                                | DEPTH_TO_SPACE                             |  Supports non-strided convolution with upscale factors of 2, 4 and 8 | 
| 17 | TIDL_SigmoidLayer              | Sigmoid/Logistic                            | SIGMOID/LOGISTIC                           |   |
| 18 | TIDL_PadLayer                  | Pad                                         | PAD                                        |   |
| 19 | TIDL_ColorConversionLayer      | NA                                          | NA                                         |  Only YUV420 NV12 format conversion to RGB/BGR color format is supported |
| 20 | TIDL_BatchReshapeLayer         | NA                                          | NA                                         |  |
| 21 | TIDL_DataConvertLayer          | NA                                          | NA                                         |  |

</div>
<br>

## Other compatible layers

<div align="center">

| No | ONNX Ops  | TFLite Ops    | Notes |
|:--:|:----------|:--------------|-------|
| 1  | Split     |               | Split layer will be removed after import |
| 2  | Reshape   | RESHAPE       | Please refer to [Meta Architecture Documentation](./tidl_fsg_od_meta_arch.md) for further details |
| 3  |            | MINIMUM       | For ReLU6 / ReLU8      |
| 4  | Transpose |               | For ShuffleChannelLayer only      |
| 5 | Clip      |               | Parametric activation threshold PACT       |

</div>
<br>

## For Unlisted Layers/Operators

Any unrecognized layers/operators will be converted to TIDL_UnsupportedLayer as a place-holder. The shape & parameters might not be correct. You may get the TIDL-RT importer result, but with such situation imported model will not work for inference on target/PC. |
<br>

# Supported model formats & operator versions
Proto files from the versions below are used for validating pre-trained models. In most cases, models from new versions should also work since the core operators tend to remain the same
  - ONNX - 1.3.0 (opset 9 and 11)
  - TFLite - Tensorflow 2.0-Alpha

*Since the Tensorflow 2.0 is planning to drop support for frozen buffer, we recommend to users to migrate to TFLite model format for Tensorflow 1.x.x as well. TFLite model format is supported in both TF 1.x.x and TF 2.x*


# Feature set comparision across devices

<div align="center">

| Feature  | AM62A | AM68A |AM68PA | AM69A|
| ------- |:-----------:|:-----------:|:-----------:|:-----------:|
| Support for native inference of TFLite PTQ Models (int8)  | :heavy_check_mark: |:heavy_check_mark: | :x: |:heavy_check_mark:|
| Support for LUT based operators  | :x: |:heavy_check_mark: | :heavy_check_mark:|:heavy_check_mark:|

</div>

*TFLite Fixed-point PTQ models will need calibration frames on (AM68PA)TDA4VM*