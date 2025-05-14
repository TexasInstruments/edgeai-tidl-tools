# Supported Operators & Runtimes

TIDL-RT supports acceleration of the operators listed below and any unsupported operator will execute on cortex-A Core as part of the corresponding  open source run time (ONNX-RT, TFLite-RT and NEO AI DLR)


<center>

## Operator Mapping (ONNX)

| S. No. | ONNX Operator | TIDL Layer | Constraints |
|:------:|:--------------|:-----------|:------|
| 1 | Conv | TIDL_ConvolutionLayer | <ul> <li> Only one variable input is allowed </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Weight tensor dimension must match the kernel_shape </li><li> Stride must be the same along both horizontal and vertical dimensions </li><li> Kernel size 3x3 with stride 3 is not supported in AM62A and AM67A </li><li> Kernel size greater than 7 with stride 2 is not supported </li><li> Depthwise (Fully Grouped) convolution is only supported for 3x3s1, 3x3s2, 5x5s2, 5x5s2, 7x7s1 & 7x7s2 filters </li><li> Padding greater than input width is not supported for AM62A and AM67A </li><li> Stride 4 is only supported with Kernel size 11x11 </li><li> Input width less than MAX(Pad Left, Pad Right) is not supported </li></ul> | 
| 2 | AveragePool/GlobalAveragePool/MaxPool | TIDL_PoolingLayer | <ul> <li> Input should be variable </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Only default dilations values (1,1) are supported </li><li> Only default storage_order = 0 is supported </li><li>In GlobalAveragePool, for plane sizes (Height * Width) larger than 1024, please use the convert_large_global_avg_pooling_to_matmul rule in [tidl-onnx-model-optimizer](../osrt-model-tools/osrt_model_tools/onnx_tools/tidl_onnx_model_optimizer/README.md)</li><li> AveragePool and MaxPool have been validated for the following kernel sizes: 3x3,2x2s,1x1 with stride 1 and stride 2 (both horizontal and vertical dimensions) </li></ul> | 
| 3 | Relu | TIDL_ReLULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 4 | PRelu | TIDL_PReLULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> PRelu does not support variable slope </li></ul> | 
| 5 | Sum/Add/Mul/Div/Sub/Max | TIDL_EltWiseLayer | <ul> <li> Constant tensor in Add/Mul/Sub/Div requires input dimensions of that layer to be present as part of the network, please run shape inference on your model</li><li> Only 2 inputs are supported in Add/Mul/Sub/Div layers </li><li> Number of non-singleton variable input dimensions in Add/Mul/Sub/Div must be less than <= 6 </li><li> The variable inputs in Add/Mul/Div/Max layer must of be same dimensions or broadcast-able </li><li> Both inputs as variable are not supported in Sub </li><li> Eltwise operator(Add/Mul/Div/Max layer) is supported only with operands of similar dimensions or broadcast supported patterns of both inputs </li><li> Constant tensor in Sub layer must be a number or 1D vector, only one dimension can be > 1 </li><li> 1D vector dimension should match with channel or width dimension </li></ul> | 
| 6 | Gemm | TIDL_InnerProductLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Filter tensor input should have atleast 2 dimensions </li><li> Bias tensor input should be a vector of [1, N] or [N] where N should match output dimension, please use [tidl-onnx-model-optimizer](../osrt-model-tools/osrt_model_tools/onnx_tools/tidl_onnx_model_optimizer/README.md) to convert Gemm to (MatMul + Add) combination otherwise </li><li> Only supported Gemm params are transA = 0, alpha = 1.0 and beta = 1.0. The same will processed as inner product or fully connected layer in TIDL </li></ul> | 
| 7 | MatMul | TIDL_InnerProductLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li> <li> Filter tensor input should have atleast 2 dimensions </li><li> MatMul with signed inputs & unsigned output is not supported  </li><li> MatMul with signed & unsigned input combination is not supported in TDA4VM </li></ul> | 
| 8 | Softmax | TIDL_SoftMaxLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only softmax along width and height axis is supported </li></ul> | 
| 9 | BatchNormalization | TIDL_BatchNormLayer | <ul> <li> training_mode = 1 is not supported </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 10 | ConvTranspose | TIDL_Deconv2DLayer | <ul> <li> Only one variable input is allowed </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Weight tensor size should match with proto kernel_shape </li><li> Only 4x4 and 2x2 kernels with 2x2 stride are supported </li><li> Only default group = 1 is supported </li><li> Change to Upsample/Resize if possible. Upsample/Resize will be more efficient </li><li> 16-bit Deconvolution is not suppported on AM62A and AM67A </li></ul> | 
| 11 | Concat | TIDL_ConcatLayer | <ul> <li> Only supported for axis values of -3, -2 & -1 </li><li> Not supported along the batch dimension </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 12 | Slice/Split | TIDL_SliceLayer | <ul> <li> Must have 4 inputs into the operator, where one is the variable (Must be <=4 dimensions) input while the other 3 are constant/initializers & 1D </li><li> Only batch size = 1 is supported </li><li> Non-one stride is not supported individually (Only supported in [Patch Merging](./tidl_fsg_vtfr.md)) </li></ul> | 
| 13 | Flatten | TIDL_FlattenLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 14 | DropOut | TIDL_DropOutLayer | <ul> <li> Not supported as an individual operator </li></ul> | 
| 15 | ArgMax | TIDL_ArgMaxLayer | <ul> <li> Only keepdims = 1 (default) is supported </li><li> Only axis = -3 is supported </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Only select_last_index = 0 (default) is supported </li></ul> | 
| 16 | Upsample/Resize | TIDL_ResizeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Only 'nearest' and 'linear' resize mode are supported </li><li> Resize is only supported along width and height axis </li><li> Scales < 1 are not supported </li><li> Width and Height scale should be same </li><li> Only Power of 2 scales are supported </li><li> Only default antialias = 0 is supported </li><li> Only default keep_aspect_ratio_policy = 'stretch' is supported </li><li> Only 'half_pixel', 'pytorch_half_pixel' and 'asymmetric' coordinate_transformation_mode is supported </li><li> Only 'round_prefer_ceil' nearest mode is supported when coordinate_transformation_mode is 'half_pixel' or 'pytorch_half_pixel' </li><li> 'pytorch_half_pixel' coordinate_transformation_mode is not supported for Resize output shapes <= 1 </li><li> Only 'floor' nearest mode is supported when coordinate_transformation_mode is 'asymmetric' </li><li> 'linear' resize mode is not supported when coordinate_transformation_mode is 'asymmetric' </li><li> Only default exclude_outside = 0 is supported </li></ul> | 
| 17 | DepthToSpace | TIDL_DepthToSpaceLayer | <ul> <li> Input should be four-dimensional (4D) </li><li> Input depth (channel dimension) should be multiple of (blocksize * blocksize) </li></ul> | 
| 18 | Sigmoid/Logistic | TIDL_SigmoidLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 19 | Pad | TIDL_PadLayer | <ul> <li> Maximum number of input dimension supported is 6 </li><li> Only constant pad mode is supported </li><li> Padding is only supported for width/height axes </li></ul> | 
| 20 | ReduceMin/ReduceMax | TIDL_ReduceLayer | <ul> <li> Reduction is only supported along height </li><li> Only keepdims = 1 is supported </li></ul> | 
| 21 | ScatterND/ScatterElements | TIDL_ScatterElementsLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> ScatterElements only supports the 'none' reduction type </li><li> ScatterND does not support the 'mul' reduction type </li><li> Updates tensor should not have more than 1 channel </li><li> Only scatter along width axis is supported </li><li> The constant input 'data' must be a zero tensor </li></ul> | 
| 22 | Squeeze | TIDL_SqueezeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 23 | Tanh | TIDL_TanhLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 24 | HardSigmoid | TIDL_HardSigmoidLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 25 | Elu | TIDL_ELULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 26 | Reshape | TIDL_ReshapeLayer | <ul> <li> Variable shape is not supported </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> allowzero is not supported </li><li> Input volume should be equal to output volume </li></ul> | 
| 27 | Gather | TIDL_GatherLayer | <ul> <li> Input dimensions must be greater than 1D </li><li> Data cannot be a constant. Only indices can be constant. </li><li> Input shape of dimension higher than axis should be 1 </li><li> Only 1D indices are supported </li></ul> | 
| 28 | Transpose | TIDL_TransposeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 29 | LayerNormalization | TIDL_LayerNormLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only supported across the width axis </li><li> Dimension of scale and bias vector can either be [1, N] or [N] </li></ul> | 
| 30 | GridSample | TIDL_GridSampleLayer | <ul> <li> Only nearest & bilinear mode is supported </li><li> Only zero padding mode is supported </li><li> Only 2D grid indices are supported </li></ul> | 
| 31 | TopK | TIDL_TopKLayer | <ul> <li> TopK is not supported with 'sorted' attribute is set to 0 </li><li> Input K for TopK operator is only supported when given as an initializer in the model </li><li> TopK axis other than height is not supported </li><li> Order of topK for same values may be different between emulation and device</li></ul> | 
| 32 | DeformConv | TIDL_DeformableConvLayer | <ul> <li>Only 3x3s1 is supported</li></ul> | 
| 33 | Clip | TIDL_ClipLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only min <= 0 and max > 0 is supported </li></ul> | 
| 34 | LeakyRelu | TIDL_LeakyReluLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 35 | Erf/Identity | TIDL_IdentityLayer | <ul> <li> Not supported as an individual operator </li></ul> | 
| 36 | DequantizeLinear | TIDL_DequantizeLayer | <ul> <li> Only default axis = 1 is supported </li><li> DeQuantizeLinear is only supported in ONNX QDQ models </li></ul> | 
| 37 | QuantizeLinear | TIDL_QuantizeLayer | <ul> <li> Only default axis = 1 is supported </li><li> QuantizeLinear is only supported in ONNX QDQ models </li></ul> | 
| 38 | Sqrt | TIDL_SqrtLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 39 | ReduceMean | TIDL_ReduceMeanLayer | <ul> <li> Not supported as an individual operator : consider using our ONNX optimizer tool to replace this layer with a supported operator configuration : https://github.com/TexasInstruments/edgeai-tidl-tools/tree/master/scripts/osrt_model_tools/onnx_tools/tidl-onnx-model-optimizer </li></ul> | 
| 40 | Pow | TIDL_PowLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 41 | Cast | TIDL_CastLayer | <ul> <li> Only supported at the terminal nodes (Input/Output) of the network </li></ul> | 
| 42 | Asin | TIDL_AsinLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 43 | Asinh | TIDL_AsinhLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 44 | HardSwish | TIDL_HardSwishLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 45 | Mish | TIDL_MishLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 46 | Log | TIDL_LogLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 47 | Unsqueeze | TIDL_UnsqueezeLayer | <ul> <li> Output dimensions after unsqueeze must be less than <= 6 </li></ul> | 
| 48 | Abs | TIDL_AbsLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 49 | Floor | TIDL_FloorLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 50 | Exp | TIDL_ExpLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 51 | Sin | TIDL_SinLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 52 | InstanceNormalization | TIDL_InstanceNormLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 53 | SpaceToDepth | TIDL_SpaceToDepthLayer | <ul> <li> Input should be four-dimensional (4D) </li><li> Input height and width should be multiple of blocksize </li></ul> | 
| 54 | Acos | TIDL_AcosLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 55 | Atan | TIDL_AtanLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 56 | Sinh | TIDL_SinhLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 57 | Neg | TIDL_NegLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 58 | Cos | TIDL_CosLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 59 | Cosh | TIDL_CoshLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 60 | Tan | TIDL_TanLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
</center>



### Support for fused combinations (ONNX)

- Certain operators are not supported individually but are parsed & supported if they form a fused combination:

<center>

| S. No. | ONNX Operator | Fused TIDL Layer | Notes |
|:------:|:--------------|:-----------|:------|
| 1 | ReduceMean | TIDL_LayerNormLayer | <ul> <li> Supported as part of the fused combination of [Layernorm](./tidl_fsg_vtfr.md). It can be individually supported by using [tidl-onnx-model-optimizer](../osrt-model-tools/osrt_model_tools/onnx_tools/tidl_onnx_model_optimizer/README.md) - however it should not be converted to MatMul if it is part of Layernorm's representation  </li></ul> | 
| 2 | Erf | TIDL_GELULayer | <ul> <li> Supported as part of the fused combination of [GELU](./tidl_fsg_vtfr.md) </li></ul> | 

</center>

<br>

<center>

## Operator Mapping (TFLite)


| S. No. | TFLite Operator | TIDL Layer | Constraints |
|:------:|:----------------|:-----------|:------|
| 1 | DepthwiseConv2d/Conv2d | TIDL_ConvolutionLayer | <ul> <li> Only one variable input is allowed </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Weight tensor dimension must match the kernel_shape </li><li> Stride must be the same along both horizontal and vertical dimensions </li><li> Kernel size 3x3 with stride 3 is not supported in AM62A and AM67A </li><li> Kernel size greater than 7 with stride 2 is not supported </li><li> Depthwise (Fully Grouped) convolution is only supported for 3x3s1, 3x3s2, 5x5s2, 5x5s2, 7x7s1 & 7x7s2 filters </li><li> Padding greater than input width is not supported for AM62A and AM67A </li><li> Stride 4 is only supported with Kernel size 11x11 </li><li> Input width less than MAX(PadL, PadR) is not supported </li></ul> | 
| 2 | AveragePool2d/Mean/MaxPool2d | TIDL_PoolingLayer | <ul> <li> Input should be variable </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Pooling has been validated for the following kernel sizes: 3x3,2x2s,1x1 with stride 1 and stride 2 (both horizontal and vertical dimensions) </li></ul> | 
| 3 | Prelu/Relu | TIDL_ReLULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 4 | Mul/Sub/Add/Div | TIDL_EltWiseLayer | <ul> <li> Constant tensor in Add/Mul/Sub/Div requires input dimensions of that layer to be present as part of the network </li><li> Only 2 inputs are supported in Add/Mul/Sub/Div layers </li><li> Number of non-singleton variable input dimensions in Add/Mul/Sub/Div must be less than <= 6 </li><li> The variable inputs in Add/Mul/Div layer must of be same dimensions or broadcast-able </li><li> Both inputs as variable are not supported in Sub </li><li> Eltwise operator(Add/Mul/Div layer) is supported only with operands of similar dimensions or broadcast supported patterns of both inputs </li><li> Constant tensor in Sub layer must be a number or 1D vector, only one dimension can be > 1 </li><li> 1D vector dimension should match with channel or width dimension </li></ul> | 
| 5 | FullyConnected | TIDL_InnerProductLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Filter tensor input should have atleast 2 dimensions </li><li> Filter and input must be of same dimensions or broadcast-able </li><li> Bias tensor input should be a vector of [1, N] or [N] where N should match output dimension </li></ul> | 
| 6 | Softmax | TIDL_SoftMaxLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 7 | Concatenation | TIDL_ConcatLayer | <ul> <li> Only supported across the width, height or channel axis </li><li> Not supported along the batch dimension </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 8 | StridedSlice | TIDL_SliceLayer | <ul> <li> Must have 4 inputs into the operator, where one is the variable (Must be <=4 dimensions) input while the other 3 are constant/initializers & 1D </li><li> Only batch size = 1 is supported </li></ul> | 
| 9 | ArgMax | TIDL_ArgMaxLayer | <ul> <li> Only axis = -3 is supported </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li></ul> | 
| 10 | ResizeBilinear/ResizeNearestNeighbor | TIDL_ResizeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li></ul> | 
| 11 | DepthToSpace | TIDL_DepthToSpaceLayer | <ul> <li> Input should be four-dimensional (4D) </li><li> Input depth (channel dimension) should be multiple of (blocksize * blocksize) </li></ul> | 
| 12 | Logistic | TIDL_SigmoidLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 13 | Pad | TIDL_PadLayer | <ul> <li> Maximum number of input dimension supported is 6 </li><li> Padding is only supported for width/height axes </li><li> Pad layer is expected to provide 8 pad values </li></ul> | 
| 14 | Quantize/Dequantize | TIDL_DataConvertLayer | <ul> </ul> | 
| 15 | Squeeze | TIDL_SqueezeLayer | <ul> </ul> | 
| 16 | Tanh | TIDL_TanhLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 17 | Elu | TIDL_ELULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 18 | Reshape | TIDL_ReshapeLayer | <ul> <li> Variable shape is not supported </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Input volume should be equal to output volume </li></ul> | 
| 19 | Transpose | TIDL_TransposeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 20 | LeakyRelu | TIDL_LeakyReluLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 21 | BatchToSpaceNd | TIDL_BatchToSpaceLayer | <ul> </ul> | 
| 22 | SpaceToBatchNd | TIDL_SpaceToBatchLayer | <ul> </ul> | 
| 23 | Pack | TIDL_PackLayer | <ul> </ul> | 
| 24 | Cast | TIDL_CastLayer | <ul> </ul> | 
| 25 | SpaceToDepth | TIDL_SpaceToDepthLayer | <ul> <li> Input should be four-dimensional (4D) </li><li> Input height and width should be multiple of blocksize </li></ul> | 


</center>

<br>

<div align="left">

# Supported model formats & operator versions
Proto files from the versions below are used for validating pre-trained models. In most cases, models from new versions should also work since the core operators tend to remain the same
  - ONNX - 1.14.0 
  - ONNX Runtime - 1.15.0 (OPSET-19)
  - TFLite - Tensorflow 2.12.0

<br>

# Feature set comparison across devices

<center>

| Feature  | AM62A | AM67A | AM68A |AM68PA | AM69A|
|:------- |:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|Support for Asymmetric, Per Channel Quantization <br> ([Asymmetric, per-axis quantization](tidl_fsg_quantization.md#d-native-support-for-tensorflow-lite-int8-ptq-models))  | :heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: | :x: |:heavy_check_mark:|
| Support for LUT accelerated non-linear activations<sup>1</sup>  | :x: | :heavy_check_mark: |:heavy_check_mark: | :heavy_check_mark:| :heavy_check_mark:|

</center>

</div>

*<sup>1</sup>LUT accelerated non-linear activations include  Sigmoid, Hard Sigmoid, GELU, TanH, Softmax & ELU*