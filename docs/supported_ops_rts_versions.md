# Supported Operators & Runtimes

TIDL-RT supports acceleration of the operators listed below and any unsupported operator will execute on cortex-A Core as part of the corresponding  open source run time (ONNX-RT, TFLite-RT and NEO AI DLR)


<center>

## Operator Mapping (ONNX)

| S. No. | ONNX Operator | TIDL Layer | Constraints |
|:------:|:--------------|:-----------|:------|
| 1 | LeakyRelu | TIDL_LeakyReluLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 2 | Cast | TIDL_CastLayer | <ul> <li> Only supported at the terminal nodes (Input/Output) of the network </li></ul> | 
| 3 | Clip | TIDL_ClipLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only min <= 0 and max > 0 is supported </li></ul> | 
| 4 | Add/Mul/Sub/Div/Sum/Max | TIDL_EltWiseLayer | <ul> <li> Constant tensor in Add/Mul/Sub/Div requires input dimensions of that layer to be present as part of the network, please run shape inference on your model </li><li> Only 2 inputs are supported in Add/Mul/Sub/Div layers </li><li> Number of non-singleton variable input dimensions in Add/Mul/Sub/Div must be less than <= 6 </li><li> The variable inputs in Add/Mul/Max layer must of be same dimensions or broadcast-able </li><li> Both inputs as variable are not supported in Sub/Div </li><li> Eltwise operator(Add/Mul/Max layer) is supported only with operands of similar dimensions or broadcast supported patterns of both inputs </li><li> Constant tensor in Sub/Div layer must be a number or 1D vector, only one dimension can be > 1 </li><li> 1D vector dimension should match with channel or width dimension </li></ul> | 
| 5 | HardSigmoid | TIDL_HardSigmoidLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 6 | Concat | TIDL_ConcatLayer | <ul> <li> Only supported for axis values of -3, -2 & -1 </li><li> Not supported along the batch dimension </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 7 | LayerNormalization | TIDL_LayerNormLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 8 | Logistic | TIDL_SigmoidLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 9 | Sigmoid | TIDL_SigmoidLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 10 | MatMul/Gemm | TIDL_InnerProductLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Filter tensor input should have atleast 2 dimensions </li><li> Bias tensor input should be a vector (1, N) and N should match output dimension </li><li> Dimension of bias vector can either be [1, N] or [N] </li><li> Only supported Gemm params are transA = 0, alpha = 1.0 and beta = 1.0. The same will processed as inner product or fully connected layer in TIDL </li><li> Gemm layer is not supported in TIDL when bias size != output size, please use [tidl-onnx-model-optimizer](../scripts/osrt_model_tools/onnx_tools/tidl_onnx_model_optimizer/README.md) to convert Gemm to (MatMul + Add) combination </li><li> MatMul with signed inputs & unsigned output is not supported  </li><li> MatMul with signed & unsigned input combination is not supported in TDA4VM & is only supported in firmware version >= 10_00_07_00 </li></ul> | 
| 11 | Tanh | TIDL_TanhLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 12 | Softmax | TIDL_SoftMaxLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only softmax along width axis is supported </li></ul> | 
| 13 | Slice/Split | TIDL_SliceLayer | <ul> <li> Must have 4 inputs into the operator, where one is the variable (Must be <=4 dimensions) input while the other 3 are constant/initializers & 1D </li><li> Only batch size = 1 is supported </li><li> Non-one stride is not supported individually (Only supported in [Patch Merging](./tidl_fsg_vtfr.md)) </li></ul> | 
| 14 | Gather | TIDL_GatherLayer | <ul> <li> Input dimensions must be greater than 1D </li><li> Number of non-singleton variable input dimensions must be less than <= 2 </li><li> Channel & higher dimensions for input should be 1 </li><li> Only line gather is supported </li><li> Data cannot be a constant. Only indices can be constant. </li></ul> | 
| 15 | Relu | TIDL_ReLULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 16 | DepthToSpace | TIDL_DepthToSpaceLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Only blocksize values of 2, 4 and 8 are supported </li><li> Standalone DepthToSpace  is not optimal unless it is next to a 1x1 convolution layer </li><li>  AM62A & AM67A currently do not support DepthToSpace  </li></ul> | 
| 17 | PRelu | TIDL_PReLULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> PRelu does not support variable slope </li></ul> | 
| 18 | GlobalAveragePool | TIDL_PoolingLayer | <ul> <li> Input should be variable </li><li>For large plane sizes being reduced, please use the convert_large_global_avg_pooling_to_matmul	rule in [tidl-onnx-model-optimizer](../scripts/osrt_model_tools/onnx_tools/tidl_onnx_model_optimizer/README.md)</li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li></ul> | 
| 19 | MaxPool | TIDL_PoolingLayer | <ul> <li> Input should be variable </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Pooling has been validated for the following kernel sizes: 3x3,2x2s,1x1 with stride 1 and stride 2 (both horizontal and vertical dimensions) </li></ul> | 
| 20 | Elu | TIDL_ELULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 21 | AveragePool | TIDL_PoolingLayer | <ul> <li> Input should be variable </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Pooling has been validated for the following kernel sizes: 3x3,2x2s,1x1 with stride 1 and stride 2 (both horizontal and vertical dimensions) </li></ul> | 
| 22 | Squeeze | TIDL_SqueezeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 23 | GridSample | TIDL_GridSampleLayer | <ul> <li> Only nearest mode is supported </li><li> Only zero padding mode is supported </li><li> Grid input should be a constant initializer in the network </li></ul> | 
| 24 | ConvTranspose | TIDL_Deconv2DLayer | <ul> <li> Only one variable input is allowed </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Weight tensor size should match with proto kernel_shape </li><li> Only 4x4, 3x3 and 2x2 kernels with 2x2 stride are supported </li><li> Change to Upsample/Resize if possible. Upsample/Resize will be more efficient </li><li> 16-bit Deconvolution is not suppported on AM62A </li></ul> | 
| 25 | BatchNormalization | TIDL_BatchNormLayer | <ul> <li> training_mode = 1 is not supported </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 26 | Reshape | TIDL_ReshapeLayer | <ul> <li> Variable shape is not supported </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> allowzero is not supported </li><li> select_last_index isn't supported </li><li> Input volume should be equal to output volume </li></ul> | 
| 27 | ArgMax | TIDL_ArgMaxLayer | <ul> <li> Only keepdims = 1 (default) is supported </li><li> Only axis = -3 is supported </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li></ul> | 
| 28 | ReduceMax | TIDL_ReduceLayer | <ul> <li> Reduction is only supported along height </li><li> Only keepdims = 1 is supported </li></ul> | 
| 29 | Resize | TIDL_ResizeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Only 'nearest' and 'linear' resize mode are supported </li><li> Only antialias value supported is zero </li><li> Only value of keep_aspect_ratio_policy supported is stretch </li></ul> | 
| 30 | Upsample | TIDL_ResizeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Only 'nearest' and 'linear' resize mode are supported </li><li> Only antialias value supported is zero </li><li> Only value of keep_aspect_ratio_policy supported is stretch </li></ul> | 
| 31 | ReduceMin | TIDL_ReduceLayer | <ul> <li> Reduction is only supported along height </li><li> Only keepdims = 1 is supported </li></ul> | 
| 32 | Conv | TIDL_ConvolutionLayer | <ul> <li> Only one variable input is allowed </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Weight tensor dimension must match the kernel_shape </li><li> Stride must be the same along both horizontal and vertical dimensions </li><li> Kernel size 3x3 with stride 3 is not supported in AM62A </li><li> Kernel size greater than 7 with stride 2 is not supported </li><li> Depthwise (Fully Grouped) convolution is only supported for 1x1s2, 3x3s1, 3x3s2, 5x5s2, 5x5s2, 7x7s1 & 7x7s2 filters </li><li> Stride 4 is only supported with Kernel size 11x11 </li><li> Input width less than MAX(Pad Left, Pad Right) is not supported </li></ul> | 
| 33 | ScatterElements | TIDL_ScatterElementsLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Only 'none' reduction is supported </li><li> Updates tensor should not have more than 1 channel </li><li> Only width direction scatter is supported </li></ul> | 
| 34 | Pad | TIDL_PadLayer | <ul> <li> Maximum number of input dimension supported is 6 </li><li> Only constant pad mode is supported </li><li> Padding is only supported for width/height axes </li></ul> | 
| 35 | ScatterND | TIDL_ScatterElementsLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Only 'none' reduction is supported </li><li> Updates tensor should not have more than 1 channel </li><li> Only width direction scatter is supported </li></ul> | 
| 36 | Transpose | TIDL_TransposeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only permutes are supported when number of dimensions > 4 </li><li> Transpose over batch dimension is not supported </li></ul> | 
| 37 | Flatten | TIDL_FlattenLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li></ul> | 
| 38 | TopK | TIDL_TopKLayer | <ul> <li> TopK is not supported with 'sorted' attribute is set to 0 </li><li> Input K for TopK operator is only supported when given as an initializer in the model </li><li> TopK axis other than height is not supported </li><li> Order of topK for same values may be different between emulation and device </li></ul> | 
| 39 | Sqrt | TIDL_SqrtLayer | <ul> </ul> |
| 40 | Sin | TIDL_SinLayer | <ul> </ul> | 
| 41 | Pow | TIDL_PowLayer | <ul> </ul> | 
| 42 | Mish | TIDL_MishLayer | <ul> </ul> | 
| 43 | Log | TIDL_LogLayer | <ul> </ul> | 
| 44 | HardSwish | TIDL_HardSwishLayer | <ul> </ul> | 
| 45 | Floor | TIDL_FloorLayer | <ul> </ul> | 
| 46 | Exp | TIDL_ExpLayer | <ul> </ul> | 
| 47 | Asinh | TIDL_AsinhLayer | <ul> </ul> | 
| 48 | Asin | TIDL_AsinLayer | <ul> </ul> | 
| 49 | Abs | TIDL_AbsLayer | <ul> </ul> | 
</center>



### Support for fused combinations (ONNX)

- Certain operators are not supported individually but are parsed & supported if they form a fused combination:

<center>

| S. No. | ONNX Operator | Fused TIDL Layer | Notes |
|:------:|:--------------|:-----------|:------|
| 1 | ReduceMean | TIDL_LayerNormLayer | <ul> <li> Supported as part of the fused combination of [Layernorm](./tidl_fsg_vtfr.md). It can be individually supported by using [tidl-onnx-model-optimizer](../scripts/osrt_model_tools/onnx_tools/tidl_onnx_model_optimizer/README.md) - however it should not be converted to MatMul if it is part of Layernorm's representation  </li></ul> | 
| 2 | Erf | TIDL_GELULayer | <ul> <li> Supported as part of the fused combination of [GELU](./tidl_fsg_vtfr.md) </li></ul> | 

</center>

<br>

<center>

## Operator Mapping (TFLite)


| S. No. | TFLite Operator | TIDL Layer | Constraints |
|:------:|:----------------|:-----------|:------|
| 1 | Pack | TIDL_PackLayer | <ul> </ul> | 
| 2 | BatchToSpaceNd | TIDL_BatchToSpaceLayer | <ul> </ul> | 
| 3 | Quantize | TIDL_DataConvertLayer | <ul> </ul> | 
| 4 | ArgMax | TIDL_ArgMaxLayer | <ul> <li> Only axis = -3 is supported </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li></ul> | 
| 5 | AveragePool2d | TIDL_PoolingLayer | <ul> <li> Input should be variable </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Pooling has been validated for the following kernel sizes: 3x3,2x2s,1x1 with stride 1 and stride 2 (both horizontal and vertical dimensions) </li></ul> | 
| 6 | Mean | TIDL_PoolingLayer | <ul> <li> Input should be variable </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Pooling has been validated for the following kernel sizes: 3x3,2x2s,1x1 with stride 1 and stride 2 (both horizontal and vertical dimensions) </li></ul> | 
| 7 | Softmax | TIDL_SoftMaxLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 8 | Prelu | TIDL_ReLULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 9 | Add/Mul/Sub/Div | TIDL_EltWiseLayer | <ul> <li> Constant tensor in Add/Mul/Sub/Div requires input dimensions of that layer to be present as part of the network </li><li> Only 2 inputs are supported in Add/Mul/Sub/Div layers </li><li> Number of non-singleton variable input dimensions in Add/Mul/Sub/Div must be less than <= 6 </li><li> The variable inputs in Add/Mul layer must of be same dimensions or broadcast-able </li><li> Both inputs as variable are not supported in Sub/Div </li><li> Eltwise operator(Add/Mul layer) is supported only with operands of similar dimensions or broadcast supported patterns of both inputs </li><li> Constant tensor in Sub/Div layer must be a number or 1D vector, only one dimension can be > 1 </li><li> 1D vector dimension should match with channel or width dimension </li></ul> | 
| 10 | Tanh | TIDL_TanhLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 11 | Pad | TIDL_PadLayer | <ul> <li> Maximum number of input dimension supported is 6 </li><li> Padding is only supported for width/height axes </li><li> Pad layer is expected to provide 8 pad values </li></ul> | 
| 12 | FullyConnected | TIDL_InnerProductLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Filter tensor input should have atleast 2 dimensions </li><li> Filter and input must of be of same dimensions or broadcast-able </li><li> Bias tensor input should be a vector (1, N) and N should match output dimension </li><li> Dimension of bias vector can either be [1, N] or [N] </li></ul> | 
| 13 | Squeeze | TIDL_SqueezeLayer | <ul> </ul> | 
| 14 | Logistic | TIDL_SigmoidLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 15 | MaxPool2d | TIDL_PoolingLayer | <ul> <li> Input should be variable </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Pooling has been validated for the following kernel sizes: 3x3,2x2s,1x1 with stride 1 and stride 2 (both horizontal and vertical dimensions) </li></ul> | 
| 16 | Relu | TIDL_ReLULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 17 | Conv2d/DepthwiseConv2d | TIDL_ConvolutionLayer | <ul> <li> Only one variable input is allowed </li><li> Number of non-singleton variable input dimensions must be less than <= 4 </li><li> Weight tensor dimension must match the kernel_shape </li><li> Stride must be the same along both horizontal and vertical dimensions </li><li> Kernel size 3x3 with stride 3 is not supported in AM62A </li><li> Kernel size greater than 7 with stride 2 is not supported </li><li> Depthwise (Fully Grouped) convolution is only supported for 3x3s1, 3x3s2, 5x5s2, 5x5s2, 7x7s1 & 7x7s2 filters </li><li> Stride 4 is only supported with Kernel size 11x11 </li><li> Input width less than MAX(PadL, PadR) is not supported </li></ul> | 
| 18 | Elu | TIDL_ELULayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 19 | Cast | TIDL_CastLayer | <ul> </ul> | 
| 20 | Dequantize | TIDL_DataConvertLayer | <ul> </ul> | 
| 21 | StridedSlice | TIDL_SliceLayer | <ul> <li> Must have 4 inputs into the operator, where one is the variable (Must be <=4 dimensions) input while the other 3 are constant/initializers & 1D </li><li> Only batch size = 1 is supported </li></ul> | 
| 22 | Reshape | TIDL_ReshapeLayer | <ul> <li> Variable shape is not supported </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Input volume should be equal to output volume </li></ul> | 
| 23 | ResizeBilinear | TIDL_ResizeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li></ul> | 
| 24 | SpaceToBatchNd | TIDL_SpaceToBatchLayer | <ul> </ul> | 
| 25 | ResizeNearestNeighbor | TIDL_ResizeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 4 </li></ul> | 
| 26 | LeakyRelu | TIDL_LeakyReluLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 
| 27 | Transpose | TIDL_TransposeLayer | <ul> <li> Number of non-singleton variable input dimensions must be less than <= 6 </li><li> Only permutes are supported when number of dimensions > 4 </li><li> Transpose over batch dimension is not supported </li></ul> | 
| 28 | Concatenation | TIDL_ConcatLayer | <ul> <li> Only supported across the width, height or channel axis </li><li> Not supported along the batch dimension </li><li> Number of non-singleton variable input dimensions must be less than <= 6 </li></ul> | 


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