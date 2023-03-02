# Supported Operators & Runtimes

# TIDL-Runtime Supported Layers Overview
1. Convolution Layer
2. Spatial Pooling Layer
    - Average and Max Pooling
3. Global Pooling Layer
    - Average and Max Pooling
4. ReLU Layer 
5. Element Wise Layer
    - Add, Product and Max
6. Inner Product Layer
    - Fully Connected Layer
7. Soft Max Layer
8. Bias Layer
9. Concatenate layer
10. Scale Layer
11. Batch Normalization layer
12. Re-size Layer (For Bi-leaner/Nearest Neighbor Up-sample)
13. RelU6 layer
14. Detection output Layer (SSD - Post Processing As defined in caffe-Jacinto and TF Object detection API)
15. Arg max layer
16. PReLU (One Parameter per channel)
17. Slice layer
18. Crop layer
19. Flatten layer
20. ShuffleChannelLayer
21. Depth to Space/ Pixel Shuffle Layer
22. Pad Layer
23. Color conversion Layer
24. Sigmoid Layer
25. Batch Reshape Layer
26. Data/Format conversion layer 

## Core Layers/Operators Mapping & Notes
| No | TIDL Layer Type                | Caffe Layer Type                    | Tensorflow Ops                          | ONNX Ops                                    | tflite Ops                                 | Notes |
|:--:|:-------------------------------|:------------------------------------|:----------------------------------------|:--------------------------------------------|:-------------------------------------------|:------|
| 1  | TIDL_ConvolutionLayer          | Convolution<br>ConvolutionDepthwise | Conv2D<br>DepthwiseConv2dNative         | Conv                                        | CONV_2D<br>DEPTHWISE_CONV_2D               | Regular & depth-wise conv will be imported as conv. <br> For TF and tflite DepthwiseConv2dNative, depth_multiplier shall be 1 in Number of input channels > 1. <br> ReLU & BN layers will be merged into conv to get better performance. 1x1 conv will be converted to innerproduct.<br>Validated kernel size: 1x1, 3x3, 5x5, 7x7,1x3,3x1,1x5,5x1,1x7,7x1.<br> If stride == 4, only supported kernel == 11x11.<br>if stride == 2, kernel should be less than 7. Even dimensions of kernel like 2x2, 4x4, 6x6 are not supported.<br>Depthwise Separable Convolution only supports 3x3,5x5,7x7 with stride 1 and 3x3 with stride 2.<br> Dilated Convolution is only supported for non-strided convolution<br> Its recommended to have kernelH*kernelW*input channel/groupNum+enableBias % 64 == 0 whereever possible as it results into better utilization of hardware.<br> **Note : Please refer MMALIB release notes for all supported configuration.**<br> **Note : Some of the kernel combination's are not optimized in current release, please refer MMALIB release notes for the same.** |
| 2  | TIDL_BatchNormLayer            | BatchNorm                           | FusedBatchNorm                          | BatchNormalization                          |                                            | ReLU & Scale & Bias & PReLU & Leaky Relu will be merged & imported as BN.<br> All the channel-wise Broad cast operations are mapped to BN now.|
| 3  | TIDL_PoolingLayer              | Pooling                             | MaxPooling<br>AvgPooling<br>Mean        | MaxPool<br>AveragePool<br>GlobalAveragePool | MAX_POOL_2D<br>AVERAGE_POOL_2D<br>MEAN     | Validated pooling size: 1x1(MAX, stride 1x1/2x2), 2x2, 3x3.<br>4x4 pooling is not optimal. |
| 4  | TIDL_EltWiseLayer              | EltWise                             | Add<br>Mul                              | Add<br>Mul                                  | ADD<br>MUL                                 | Only support SUM/MAX/PRODUCT.<br>Only support 2 inputs. |
| 5  | TIDL_InnerProductLayer         | InnerProduct                        | MatMul                                  | Gemm                                        | FULLY_CONNECTED                            | Input shape must be 1x1x1xN. Please use global pooling/flatten before innerproduct.<br>Feature size larger than 2048*2048 is not optimal. |
| 6  | TIDL_SoftMaxLayer              | SoftMax                             | Softmax                                 | Softmax                                     | SOFTMAX                                    | Input shape must be 1x1x1xN. Please use global pooling/flatten before softmax. |
| 7  | TIDL_Deconv2DLayer             | Deconvolution                       | Conv2DTranspose                         | ConvTranspose                               | TRANSPOSE_CONV                             | Only 8x8, 4x4 and 2x2 kernel with 2x2 stride is supported. Recommend to use Resize/Upsample to get better performance. The output feature-map size shall be 2x the input|
| 8  | TIDL_ConcatLayer               | Concat                              | ConcatV2                                | Concat                                      | CONCATENATION                              | Concat will do channel-wise combination by default. Concat will be width-wise if coming after a flatten layer. used in the context of SSD.<br> Width/Height wise concat is supported with Caffe|
| 9  | TIDL_SliceLayer                | Slice                               | Slice                                   | Split                                       | NA                                         | Only support channel-wise slice. |
| 10 | TIDL_CropLayer                 | Crop                                | NA                                      | NA                                          | NA                                         |  |
| 11 | TIDL_FlattenLayer              | Flatten                             | NA                                      | Flatten                                     | NA                                         | 16bit is not optimal in current version. |
| 12 | TIDL_ArgMaxLayer               | ArgMax                              | Argmax                                  | ArgMax                                      | ARG_MAX                                    | Only support axis == 1, mainly for the last layer of sematic segmentation. |
| 13 | TIDL_DetectionOutputLayer      | DetectionOutput                     | tensorflow Object Detection API         | NA                                          | NA                                         | Please refer to comment 1. |
| 14 | TIDL_ShuffleChannelLayer       | ShuffleChannel                      | NA                                      | Reshape + Transpose + Reshape               | NA                                         |  |
| 15 | TIDL_ResizeLayer               | NA                                  | ResizeNearestNeighbor<br>ResizeBilinear | UpSample                                    | RESIZE_NEAREST_NEIGHBOR<br>RESIZE_BILINEAR | Only support Power of 2 and symmetric resize. Note that any resize ratio which is power of 2 and greater than 4 will be placed by combination of 4x4 resize layer and 2x2 resize layer. As an example a 8x8 resize will be replaced by 4x4 resize followed by 2x2 resize  |
| 16 | TIDL_DepthToSpaceLayer          | NA                                  | NA | DepthToSpace                      | DEPTH_TO_SPACE |  Supports non-strided convolution with upscale of 2, 4 and 8 | 
| 17 | TIDL_SigmoidLayer          | SIGMOID/LOGISTIC                         | Sigmoid/Logistic | Sigmoid/Logistic                       | SIGMOID/LOGISTIC |   |
| 18 | TIDL_PadLayer          | NA                                  | Pad | Pad                                    | PAD |   |
| 19 | TIDL_ColorConversionLayer          | NA                                  | NA | NA                                    | NA |  Only YUV420 NV12 format conversion to RGB/BGR color format is supported |
| 20 | TIDL_BatchReshapeLayer | NA                                  | NA | NA                                    | NA |  used to covert batch of images to format which suits TIDL-RT and then convert back, refer [here](tidl_fsg_batch_processing.md) for more details |
| 21 | TIDL_DataConvertLayer          | NA                                  | NA | NA                                    | NA |  NA |

## Other compatible layers
| No | Caffe Layer Type | Tensorflow Ops | ONNX Ops  | tflite Ops    | Notes |
|:--:|:-----------------|:---------------|:----------|:--------------|-------|
| 1  | Bias             | BiasAdd        |           |               | Bias will be imported as BN. |
| 2  | Scale            |                |           |               | Scale will be imported as BN. |
| 3  | ReLU             | Relu<br>Relu6  | Relu      | RELU<br>RELU6 | ReLU will be imported as BN. |
| 4  | PReLU            |                | Prelu     |               | PReLU will be imported as BN. |
| 5  | Split            |                | Split     |               | Split layer will be removed after import. |
| 6  | Reshape          | Reshape        | Reshape   | RESHAPE       | Please refer to comment 1. |
| 7  | Permute          |                |           |               | Please refer to comment 1. |
| 8  | Priorbox         |                |           |               | Please refer to comment 1. |
| 9  |                  | Pad            | Pad       | PAD           | Padding will be taken care of during import process, and this layer will be automatically removed by import tool. |
| 10 |                  |                |           | MINIMUM       | For relu6 / relu8      |
| 11 | DropOut          |                |           |               | This layer is only used in training, and this layer will be automatically removed during import process. |
| 12 |                  | Squeeze        |           |               | Flatten       |
| 13 |                  | Shape          |           |               | Resize       |
| 14 |                  |                | Transpose |               | For ShuffleChannelLayer       |
| 15 |                  |                | Clip      |               | Parametric activation threshold PACT       |
| 16 |                  |                | LeakyRelu |  LEAKY_RELU   | Leaky Relu will be imported as BN       |



## For Unlisted Layers/Operators

Any unrecognized layers/operators will be converted to TIDL_UnsupportedLayer as a place-holder. The shape & parameters might not be correct. You may get the TIDL-RT importer result, but with such situation imported model will not work for inference on target/PC. |

* If this operation is supported by TIDL-RT inference, but not supported by TIDL-RT import tool:
    - Please modify the import tool source code.
* If this operation is not supported by TIDL-RT inference, use open source run time

# Supported model formats & operator versions
Proto file from below version are used for validating pre-trained models. In most cases new version models also shall work since the basic operations like convolution, pooling etc don't change
  - Caffe - 0.17 (caffe-jacinto in gitHub)
  - Tensorflow - 1.12
  - ONNX - 1.3.0 (opset 9 and 11)
  - TFLite - Tensorflow 2.0-Alpha

*Since the Tensorflow 2.0 is planning to drop support for frozen buffer, we recommend to users to migrate to TFlite model format for Tensorflow 1.x.x as well. TFLite model format is supported in both TF 1.x.x and TF 2.x*

*Fixed-point models are only supported for TFLite & need calibrations images for TDA4VM"

#Feature set comparison across devices:
