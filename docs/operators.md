# Supported Operators

TIDL-RT supports acceleration of the operators listed below and any unsupported operator will execute on cortex-A Core as part of the corresponding runtime.

> **Note:** Test reports for various operator with varying attributes and shapes is published foe every release of edgeai-tidl-tools, please find the reports [here](../test/reports/).

<!-- TOC -->
   - [ONNX](#onnx)
      - [Version](#version)
      - [Operators Supported](#operators-supported)
   - [TFLite](#tflite)
      - [Version](#version-1)
      - [Operators Supported](#operators-supported-1)
<!-- /TOC -->

## ONNX

### Version
  - ONNX - 1.14.0 
  - ONNX Runtime - 1.23.0 ([OPSET-21 IR-10](https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support))

### Operators Supported

| S. No. | Onnx Operator | TIDL Layer | Constraints | Notes |
|:------:|:--------------|:-----------|:------|:------|
| 1 | Conv | TIDL_ConvolutionLayer | <ul> <li> Only one variable input is allowed </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Weight tensor dimension must match the kernel_shape </li><li> Stride must be the same along both horizontal and vertical dimensions </li><li> Kernel size 3x3 with stride 3 is not supported in AM62A and AM67A </li><li> Kernel size greater than 7 with stride 2 is not supported </li><li> Depthwise (Fully Grouped) convolution is only supported for 1x3s1, 3x3s1, 3x3s2, 5x5s1, 5x5s2, 7x7s1 & 7x7s2 filters </li><li> Padding greater than input width is not supported for AM62A and AM67A </li><li> Stride 4 is only supported with Kernel size 11x11 </li><li> Input width less than MAX(Pad Left, Pad Right) is not supported for AM62A and AM67A </li></ul> | |
| 2 | AveragePool/GlobalAveragePool/MaxPool | TIDL_PoolingLayer | <ul> <li> Input should be variable </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> GlobalAveragePool cannot support plane sizes > 1024. You can use the convert_large_global_avg_pooling_to_matmul rule in tidl-onnx-model-optimizer. </li><li> Only 1D and 2D GlobalAveragePool is supported </li><li> Pooling functionality is validated for 3x3,2x2 and 1x1 kernels with both stride 1 and stride 2 (horizontally and vertically), with the exception of 2x2 kernel with stride 1 </li><li> Only default dilations values (1,1) are supported </li><li> Number of outputs should be 1 </li></ul> | <ul><li>For GlobalAveragePool, plane sizes (height*width) > 1024,  please use the convert_large_global_avg_pooling_to_matmul rule in [tidl-onnx-model-optimizer](../osrt-model-tools/osrt_model_tools/onnx_tools/tidl_onnx_model_optimizer/README.md)</li></ul> |
| 3 | Relu | TIDL_ReLULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 4 | PRelu | TIDL_PReLULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> PRelu does not support variable slope </li><li> PRelu slope should be same or broadcast-able to input channel dimension </li></ul> |  |
| 5 | Max/Sum/Add/Mul/Div/Sub/Min | TIDL_EltWiseLayer | <ul> <li> Only 2 inputs are supported in Add/Mul/Sub/Div/Max/Min layers </li><li> Number of non-singleton variable input dimensions in Sum/Add/Mul/Sub/Div/Max must be less than <= 6 </li><li> The variable inputs in Add/Mul/Div/Sub/Max/Min layer must of be same dimensions or broadcast-able </li><li> Eltwise operator(Add/Mul/Div/Sub/Max/Min layer) is supported only with operands of similar dimensions or broadcast supported patterns of both inputs </li><li> 1D vector dimension should match with channel or width dimension </li></ul> | <ul><li>Constant tensor requires input dimensions of that layer to be present as part of the network, please run shape inference on your model</li></ul>  |
| 6 | Gemm/MatMul | TIDL_InnerProductLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Filter tensor input should have atleast 2 dimensions </li><li> Gemm layer does not support more than one variable inputs </li><li> Bias tensor input should be a vector of [1, N] or [N] where N should match output dimension </li><li> Only supported Gemm params are transA = 0, alpha = 1.0 and beta = 1.0. The same will processed as inner product or fully connected layer in TIDL </li><li> Gemm layer is not supported in TIDL when bias size != output width </li><li> MatMul with signed inputs & unsigned output is not supported  </li><li> MatMul with signed & unsigned input combination is not supported in TDA4VM</li></ul> |  <ul><li>For Gemm, if bias tensor is not a vector of [1, N] or [N] (where N is the output width), please use [tidl-onnx-model-optimizer](../osrt-model-tools/osrt_model_tools/onnx_tools/tidl_onnx_model_optimizer/README.md) to convert Gemm to (MatMul + Add) combination using convert_gemm_to_matmul_and_add rule </li></ul> |
| 7 | Softmax | TIDL_SoftMaxLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only softmax along width and height axis is supported </li></ul> |  |
| 8 | BatchNormalization | TIDL_BatchNormLayer | <ul> <li> Number of variable input dimensions must be less than 6 </li><li> training_mode = 1 is not supported </li><li> Scale, Bias, input_mean and input_var should be constant 1-D tensor of size same as input channel dimension </li></ul> |  |
| 9 | ConvTranspose | TIDL_Deconv2DLayer | <ul> <li> Only one variable input is allowed </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Weight tensor size should match with proto kernel_shape </li><li> Only 4x4, 3x3 and 2x2 kernels with 2x2 stride are supported </li><li> Change to Upsample/Resize if possible. Upsample/Resize will be more efficient </li><li> 16-bit Deconvolution is not suppported on AM62A and AM67A </li><li> Only default dilations values (1,1) are supported </li></ul> |  |
| 10 | Concat | TIDL_ConcatLayer | <ul> <li> Only supported for axis values of -3, -2 & -1 </li><li> Not supported along the batch dimension </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 11 | Slice/Split | TIDL_SliceLayer | <ul> <li> Must have 4 inputs </li><li> Only one input should be variable </li><li> Should have 3 constant inputs </li><li> Number of dimensions for variable inputs must be 4 </li><li> Number of dimensions for constant inputs must be 1 </li><li> Constant inputs must have 4 values </li><li> Number of dimensions for output tensor must be 4 </li><li> Only batch size = 1 is supported </li><li> Only supports non-batch dimension </li><li> Non-singular stride are not supported individually </li></ul> | <ul><li>Non-singular stride are not supported individually and is only supported if the Slice/Split is a part of [Patch Merging](./vision_transformers.md#patch-merging) fusion pattern </li></ul>  |
| 12 | Flatten | TIDL_FlattenLayer | <ul> <li> Number of non-singleton variable input dimensions must be <= 6 </li></ul> |  |
| 13 | DropOut | TIDL_DropOutLayer | <ul> <li> Not supported as an individual operator </li></ul> |  |
| 14 | ArgMin/ArgMax | TIDL_ArgOpLayer | <ul> <li> Only keepdims = 1 (default) is supported </li><li> Only axis = -3 is supported </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> select_last_index isn't supported </li></ul> |  |
| 15 | Upsample/Resize | TIDL_ResizeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Only 'nearest' and 'linear' resize mode are supported </li><li> Resize is only supported along width and height axis </li><li> Scales < 1 are not supported </li><li> Width and Height scale should be same </li><li> Only Power of 2 scales are supported </li><li> Only default antialias = 0 is supported </li><li> Only default keep_aspect_ratio_policy = 'stretch' is supported </li><li> Only 'half_pixel', 'pytorch_half_pixel' and 'asymmetric' coordinate_transformation_mode is supported </li><li> Only 'round_prefer_ceil' nearest mode is supported when coordinate_transformation_mode is 'half_pixel' or 'pytorch_half_pixel' </li><li> 'pytorch_half_pixel' coordinate_transformation_mode is not supported for Resize output shapes <= 1 </li><li> Only 'floor' nearest mode is supported when coordinate_transformation_mode is 'asymmetric' </li><li> 'linear' resize mode is not supported when coordinate_transformation_mode is 'asymmetric' </li><li> Only default exclude_outside = 0 is supported </li></ul> |  |
| 16 | DepthToSpace | TIDL_DepthToSpaceLayer | <ul> <li> Input should be four-dimensional (4D) </li><li> Input depth (channel dimension) should be multiple of (blocksize * blocksize) </li><li> Only mode supported are DCR(depth-column-row order) and CRD(column-row-depth order) </li></ul> |  |
| 17 | Sigmoid/Logistic | TIDL_SigmoidLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 18 | Pad | TIDL_PadLayer | <ul> <li> Maximum number of input dimension supported is 6 </li><li> Only constant pad mode with constant_value = 0 is supported </li><li> Padding is only supported for width/height axes </li></ul> |  |
| 19 | ReduceMin/ReduceMax | TIDL_ReduceLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Reduction is only supported along height axis </li><li> Only keepdims = 1 is supported </li></ul> |  |
| 20 | ScatterND/ScatterElements | TIDL_ScatterElementsLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> ScatterElements only supports the 'none' reduction type </li><li> ScatterND does not support the 'mul' reduction type </li><li> Updates tensor should not have more than 1 channel </li><li> Only scatter along width axis is supported </li><li> For ScatterND, indices tensor's last dimension must be at most (output_rank - 1) </li><li> The constant input 'data' must be a zero tensor </li></ul> |  |
| 21 | Squeeze | TIDL_SqueezeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 22 | Tanh | TIDL_TanhLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 23 | HardSigmoid | TIDL_HardSigmoidLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 24 | Elu | TIDL_ELULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 25 | Reshape | TIDL_ReshapeLayer | <ul> <li> Variable shape is not supported </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only default allowzero = 0 is supported </li><li> Input volume should be equal to output volume </li></ul> |  |
| 26 | Gather | TIDL_GatherLayer | <ul> <li> Input dimensions must be greater than 1D </li><li> Number of output dimensions must be less than <= 6 </li><li> Data cannot be a constant. Only indices can be constant. </li><li> Input shape of dimension higher than axis should be 1 </li><li> Only 1D indices are supported </li></ul> |  |
| 27 | Transpose | TIDL_TransposeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Transpose: For inputs with more than 4 dimensions (>4D), permutations where the width dimension is mapped to either the first or second position (Dim 0 or Dim 1) in the output shape are not supported.Unsupported permutation patterns:[W, X, X, X, X, X] - Width dimension at position 0; [X, W, X, X, X, X] - Width dimension at position 1 </li></ul> |  |
| 28 | LayerNormalization | TIDL_LayerNormLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only supported across the width axis </li><li> Dimension of scale and bias vector can either be [1, N] or [N] </li></ul> |  |
| 29 | GridSample | TIDL_GridSampleLayer | <ul> <li> Only nearest & bilinear mode is supported </li><li> Only zero padding mode is supported </li><li> Only 2D grid indices are supported </li></ul> | <ul><li>The rounding behaviour is different for nearest-neighbour mode for TIDL compared to ONNX, which might lead to degradation in output </li></ul> |
| 30 | TopK | TIDL_TopKLayer | <ul> <li> TopK is not supported with 'sorted' attribute is set to 0 </li><li> Input K for TopK operator is only supported when given as an initializer in the model </li><li> TopK axis other than width and height is not supported </li></ul> | <ul><li>Order of TopK for same values may be different between host emulation and device runs</li><li> TIDL has a custom implementntaion of TopK's output buffer. Refer to [TopK](./topk.md) for information.</li></ul> |
| 31 | DeformConv | TIDL_DeformableConvLayer | <ul> <li> DeformConv is only supported for 3x3s1 filter </li></ul> |  |
| 32 | Clip | TIDL_ClipLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only min <= 0 and max > 0 is supported </li></ul> |  |
| 33 | LeakyRelu | TIDL_LeakyReluLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 34 | Erf/Identity | TIDL_IdentityLayer | <ul> <li> Not supported as an individual operator </li></ul> | <ul> <li> Only supported as part of the fused combination of [GELU](./vision_transformers.md#gelu)</li></ul> |
| 35 | DequantizeLinear | TIDL_DequantizeLayer | <ul> <li> Only default axis = 1 is supported </li><li> DeQuantizeLinear is only supported in ONNX QDQ models </li></ul> |  |
| 36 | QuantizeLinear | TIDL_QuantizeLayer | <ul> <li> Only default axis = 1 is supported </li><li> QuantizeLinear is only supported in ONNX QDQ models </li></ul> |  |
| 37 | Sqrt | TIDL_SqrtLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 38 | ReduceMean | TIDL_ReduceMeanLayer | <ul> </ul> |  |
| 39 | ReduceSum | TIDL_ReduceSumLayer | <ul> </ul> |  |
| 40 | Pow | TIDL_PowLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Power tensor must be a constant tensor </li><li> Size of the constant tensor must be 1 </li></ul> |  |
| 41 | Cast | TIDL_CastLayer | <ul> <li> Only supported at the terminal nodes (Input/Output) of the network </li></ul> |  |
| 42 | Asin | TIDL_AsinLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 43 | Asinh | TIDL_AsinhLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 44 | HardSwish | TIDL_HardSwishLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 45 | Mish | TIDL_MishLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 46 | Log | TIDL_LogLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 47 | Unsqueeze | TIDL_UnsqueezeLayer | <ul> <li> Output dimensions after unsqueeze must be less than <= 6 </li></ul> |  |
| 48 | Abs | TIDL_AbsLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 49 | Floor | TIDL_FloorLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 50 | Exp | TIDL_ExpLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 51 | Sin | TIDL_SinLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 52 | InstanceNormalization | TIDL_InstanceNormLayer | <ul> <li> Number of non-singleton variable input dimensions must be <= 6 </li><li> Number of variable input dimensions must be >= 3 </li><li> Scale and Bias should be 1-D tensor of size same as input channel dimension </li></ul> |  |
| 53 | SpaceToDepth | TIDL_SpaceToDepthLayer | <ul> <li> Input should be four-dimensional (4D) </li><li> Input height and width should be multiple of blocksize </li></ul> |  |
| 54 | Acos | TIDL_AcosLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 55 | Atan | TIDL_AtanLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 56 | Sinh | TIDL_SinhLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 57 | Neg | TIDL_NegLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 58 | Cos | TIDL_CosLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 59 | Cosh | TIDL_CoshLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 60 | Tan | TIDL_TanLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 61 | Expand | TIDL_ExpandLayer | <ul> <li> Number of non-singleton variable input dimensions in Sum/Add/Mul/Sub/Div/Max must be less than <= 6 </li><li> Shape tensor cannot be a variable input </li></ul> | |

## TFLite

### Version
  - Tensorflow - 2.12.0

## Operators Supported
| S. No. | TFLite Operator | TIDL Layer | Constraints | Notes |
|:------:|:----------------|:-----------|:------|:------|
| 1 | DepthwiseConv2d/Conv2d | TIDL_ConvolutionLayer | <ul> <li> Only one variable input is allowed </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Weight tensor dimension must match the kernel_shape </li><li> Stride must be the same along both horizontal and vertical dimensions </li><li> Kernel size 3x3 with stride 3 is not supported in AM62A and AM67A </li><li> Kernel size greater than 7 with stride 2 is not supported </li><li> Depthwise (Fully Grouped) convolution is only supported for 1x3s1, 3x3s1, 3x3s2, 5x5s1, 5x5s2, 7x7s1 & 7x7s2 filters </li><li> Padding greater than input width is not supported for AM62A and AM67A </li><li> Stride 4 is only supported with Kernel size 11x11 </li><li> Input width less than MAX(PadL, PadR) is not supported for AM62A and AM67A </li></ul> | |
| 2 | AveragePool2d/Mean/MaxPool2d | TIDL_PoolingLayer | <ul> <li> Input should be variable </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> GlobalAveragePool cannot support plane sizes > 1024. You can use the convert_large_global_avg_pooling_to_matmul rule in tidl-onnx-model-optimizer. </li><li> Only 1D and 2D GlobalAveragePool is supported </li><li> Pooling functionality is validated for 3x3,2x2 and 1x1 kernels with both stride 1 and stride 2 (horizontally and vertically), with the exception of 2x2 kernel with stride 1 </li></ul> |  |
| 3 | Prelu/Relu | TIDL_ReLULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 4 | Mul/Add/Sub/Div | TIDL_EltWiseLayer | <ul> <li> Only 2 inputs are supported in Add/Mul/Sub/Div layers </li><li> Number of non-singleton variable input dimensions in Add/Mul/Sub/Div must be less than <= 6 </li><li> The variable inputs in Add/Mul/Div/Sub layer must of be same dimensions or broadcast-able </li><li> Eltwise operator(Add/Mul/Div/Sub layer) is supported only with operands of similar dimensions or broadcast supported patterns of both inputs </li><li> 1D vector dimension should match with channel or width dimension </li></ul> |  |
| 5 | FullyConnected | TIDL_InnerProductLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Filter tensor input should have atleast 2 dimensions </li><li> Filter and input must be of same dimensions or broadcast-able </li><li> Bias tensor input should be a vector of [1, N] or [N] where N should match output dimension </li></ul> |  |
| 6 | Softmax | TIDL_SoftMaxLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 7 | TransposeConv | TIDL_Deconv2DLayer | <ul> <li> Only one variable input is allowed </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Weight tensor size should match with proto kernel_shape </li><li> Only 4x4, 3x3 and 2x2 kernels with 2x2 stride are supported </li><li> Change to Resize if possible. Resize will be more efficient </li><li> 16-bit Deconvolution is not suppported on AM62A and AM67A </li><li> Only default dilations values (1,1) are supported </li></ul> |  |
| 8 | Concatenation | TIDL_ConcatLayer | <ul> <li> Only supported across the width, height or channel axis </li><li> Not supported along the batch dimension </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 9 | StridedSlice | TIDL_SliceLayer | <ul> <li> Must have 4 inputs </li><li> Only one input should be variable </li><li> Should have 3 constant inputs </li><li> Number of dimensions for variable inputs must be 4 </li><li> Number of dimensions for constant inputs must be 1 </li><li> Constant inputs must have 4 values </li><li> Number of dimensions for output tensor must be 4 </li><li> Only batch size = 1 is supported </li><li> Only supports non-batch dimension </li><li> Non-singular stride are not supported individually </li></ul> |  |
| 10 | ArgMax | TIDL_ArgOpLayer | <ul> <li> Only axis = -3 is supported </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li></ul> |  |
| 11 | ResizeBilinear/ResizeNearestNeighbor | TIDL_ResizeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li></ul> |  |
| 12 | DepthToSpace | TIDL_DepthToSpaceLayer | <ul> <li> Input should be four-dimensional (4D) </li><li> Input depth (channel dimension) should be multiple of (blocksize * blocksize) </li><li> Only mode supported are DCR(depth-column-row order) and CRD(column-row-depth order) </li></ul> |  |
| 13 | Logistic | TIDL_SigmoidLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 14 | Pad | TIDL_PadLayer | <ul> <li> Maximum number of input dimension supported is 6 </li><li> Padding is only supported for width/height axes </li><li> Pad layer is expected to provide 8 pad values </li></ul> |  |
| 15 | Quantize/Dequantize | TIDL_DataConvertLayer | <ul> </ul> |  |
| 16 | Squeeze | TIDL_SqueezeLayer | <ul> </ul> |  |
| 17 | Tanh | TIDL_TanhLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 18 | Elu | TIDL_ELULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 19 | Reshape | TIDL_ReshapeLayer | <ul> <li> Variable shape is not supported </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Input volume should be equal to output volume </li></ul> |  |
| 20 | Transpose | TIDL_TransposeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Transpose: For inputs with more than 4 dimensions (>4D), permutations where the width dimension is mapped to either the first or second position (Dim 0 or Dim 1) in the output shape are not supported.Unsupported permutation patterns:[W, X, X, X, X, X] - Width dimension at position 0; [X, W, X, X, X, X] - Width dimension at position 1 </li></ul> |  |
| 21 | LeakyRelu | TIDL_LeakyReluLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> |  |
| 22 | BatchToSpaceNd | TIDL_BatchToSpaceLayer | <ul> </ul> |  |
| 23 | SpaceToBatchNd | TIDL_SpaceToBatchLayer | <ul> </ul> |  |
| 24 | Pack | TIDL_PackLayer | <ul> </ul> |  |
| 25 | Cast | TIDL_CastLayer | <ul> </ul> |  |
| 26 | SpaceToDepth | TIDL_SpaceToDepthLayer | <ul> <li> Input should be four-dimensional (4D) </li><li> Input height and width should be multiple of blocksize </li></ul> |  |
