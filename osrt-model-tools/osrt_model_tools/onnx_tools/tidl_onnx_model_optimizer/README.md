# tidl-onnx-model-optimizer
This module contains various modifications and optimization on the original ONNX graph and generates a new optimized ONNX model with the same functionality as the original model but with a more optimized structure for TIDL-RT inference.

## Setup
For setting up execute the command

    cd ../onnx_tools
    source ./setup.sh

This depends on the [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon)

## Use
After running setup script, you will have `tidl_onnx_model_optimizer` python package installed in your python environment, which can be used as follows:

    from tidl_onnx_model_optimizer import optimize

    optimize(input_model_path, output_model_path, **kwargs)

Input arguments to `optimize`

    model:                  path to input ONNX model
    out_model:              path to output ONNX model (optional).
                            If not given, saved in same place as the input model
                            with a default name (optimized_<input_model_name>)
    shape_inference_mode:   (pre/post/all/None) flag to use onnx shape inference
                            [pre: run only before graph surgeon optimization,
                            post:run only after graph surgeon optimization,
                            all (default): both pre and post are enabled,
                            None: both disabled]
    simplify_mode:          (pre/post/all/None) flag to use onnxsim simplification
                            [pre : simplify only before graph surgeon
                            optimizations, post:simplify only after graph
                            surgeon optimization, all: both pre and post are
                            enabled, None (default): both disabled]


The above command will generate the output model in the same location as mentioned. Some shapes are removed while making changes to graph structure, so shape inference has to be run on the output model before model compilation


## Operations
The different optimizations performed are summarized here along with their default flag value (Enabled = True, Disabled = False).

| Sl. no. | Function (flag)                       | Summary                               |       Default     |
|:------: | :------------------------------------ |:------------------------------------: | :---------------- |
| 1 | convert_resize_params_size_to_scale | Resize operator can specify either size of scale parameter in input, but TIDL does not support size input params. This function converts size to corresposding scale. For e.g, with input [3, 256, 256] and size input [3, 128, 128], it will convert to scales [1, 2, 2]| False |
| 2 | convert_concat_axis_width_to_channel | TIDL only supports concat on channel axis. This function converts Concat layer with width axis to Concat layer with channel axis adjusting the input and output accordingly with Reshapes | False |
| 3 | convert_maxpool_to_cascaded_maxpool | The MaxPool layer with large kernel (> 3x3) is replaced with cascaded MaxPool layers wiht 3x3 kernel. Assume that the kernel size is NxN where N is odd | True |
| 4 | convert_reducemean_to_matmul | The ReduceMean layer is replaced with the cascaded multiple layers, e.g., "Reshape + MatMul + Reshape". The attribute, "axes" of ReduceMean should be W and H dimension. ReduceMean in channel dimension is not supported | True |
| 5 | convert_gemm_to_matmul_and_add | Gemm layer with constant B input in converted to Matmul and Gemm bias (if exists) is converted to a following Add layer | False |
| 6 | convert_matmul_to_conv_1x1s1 | Function to convert MatMul layer to Convolution with kernel 1x1, stride 1x1. Only works for MatMuls with input dimensions not equal to 3 (i.e., 2 or >= 4 works) | False |
| 7 | convert_large_global_avg_pooling_to_matmul | Global average pooling with large HxW values might be unoptimal, converting the input with a reshape from HxW to 1xHW and doing MatMul with a const tensor of dim HWx1 and value of 1/HW | True |
| 8 | convert_gather_with_single_index_to_slice | Gather layer with a single index = t, can be converted to Slice [t, t+1] on the same axis | True |
| 9 | convert_batchnorm_input_to_4D |  Batchnorm input with less than 4 dimension is converted to 4 dimension by adding 1's at the end, done using Reshaped before and after the layer. TIDL supports only 4D batchnorm (NCHW) with batchnorm on the channel | True |
| 10 | attention_block_optimization | Attention block optimization function, identifies attention blocks and performs TIDL specific optimizations on the attention blocks as a whole | False |
| 11 | split_batch_dim_to_parallel_input_branches | If network has batch dimensions to some layers which does not suppport batch dim in TIDL framework, duplicate the layer and split in multiple branches so as each batch gets treated as different input to different branch | False |
| 12 | convert_softmax_axis_height_to_width | The SoftMax layer with operation in the height dimension is replaced with Transpose -> SoftMax -> Transpose to satisfy constraint of SoftMax layer only occuring in width dimension | True |
| 13 | convert_softmax_axis_channel_to_width | The SoftMax layer with operation in the channel dimension is replaced with Transpose -> SoftMax -> Transpose to satisfy constraint of SoftMax layer only occuring in width dimension | True |
| 14 | push_large_channel_dim_to_height_for_width_wise_softmax | When a softmax has high value of dimensions channel and upper it performs unoptimal. But reshaping the shape to have a larger height can make it more efficient. Hence Softmax is changed to Reshape -> Softmax -> Reshape | True |
| 15 | convert_conv_large_pad_to_smaller_kernel | Convolution layer with large kernels and small inputs might be unsupported when pad is greater than the input dimension. This can be converted to Conv with smaller kernel and less pad for support | True |
| 16 | expand_layernorm_to_component_ops | The LayerNormalization-17 layer from ONNX is not supported by TIDL. We can expand this layer to it's fundamental operators to make it supported in TIDL | False |
| 17 | push_matmul_channel_in_height | Matmul layers with one input broadcasted across channel and other input with small plane size can have the channel and height axis merged to get optimized performance | False |
| 18 | expand_slice_across_multiple_axis | Slice along a single axis is currently supported for TIDL import. This will split the slice into multiple slices each acting on a single axis. | True |
| 19 | convert_instancenorm_to_layernorm | InstanceNormalisation is not supported in TIDL, converting it to LayerNorm with the same functionality. | False |
| 20 | convert_unsqueeze_to_reshape | Converts the Unsqueeze layer to reshape layer for support. | False |
| 21 | add_bias_qdq | Adds the bias quantization to conv layers if not already there (Weight_params * Act_params) | False |
| 22 | remove_quantize_initializer | Removes the Quantization node in initialisers (reduces the model size as input becomes 8-bit) - Use only for PT2E exported models (quantization=3) | True |
| 23 | remove_duplicate_quantize_dequantize | Removes the duplicate sequential Q-DQ layers (keeps the first quant params) | False |
| 24 | convert_neg_to_mul | Converts the Neg layer (from RoPE) to mul by -1 | True |
| 25 | convert_expand_to_reshape_and_concat | Converts the expand layer to reshape and concat | True |
| 26 | convert_single_concat_to_consecutive_concats | Convert a concat which works as expanding a dimension of a tensor (1x1x10 -> 1x5x10) to multiple consecutive concats which only takes 2 inputs at once, thus in the example, we would have 4 different concats.  | True |
| 27 | convert_conv_7x7_stride4_to_stride1 | Few models(segformer) has a convolution layer with 7x7 kernel and 4 stride, converting the layer to the one with a stride of 1 using combination of maxpool and conv  | True |
| 28 | convert_2_dimension_slice_to_maxpool | Slice if present in 2 axes, with same steps, it is converted to a corresponding maxpool with kernel size of 1, transpose also are inserted if channel not in 2nd dimension | False |
| 29 | Change_argmax_keepdims_to_1 | Changes the keepdims parameter from 0 to 1 and adds a reshape node to accomodate for the shape | False |
| 30 | Hf_attention_block_optimization | Attention block optimization function, identifies attention blocks and performs TIDL specific optimizations on the attention blocks as a whole | True |
| 31 | convert_reducesum_to_matmul | The ReduceSum layer is replaced with the cascaded multiple layers, e.g., "Reshape + MatMul + Reshape". The attribute, "axes" of ReduceSum should be W and H dimension. ReduceSum in channel dimension is not supported | True |
| 32 | convert_resize_params_size_to_scale_dynamic_batch | Finds Resize nodes that use a sequence of nodes that dynamically determine output sizes, which are added during export. The rule determines the static 'scales', and removes the dynamic nodes such that the Reize node is supported | False |
| 33 | replace_mean_with_eltwise | Replaced Mean of 2 tensors with Add + Multiply by 0.5 (since Div is also not supported). >2 inputs is not supported, but could be implemented without much difficulty | False |
| 34 | replace_sub_with_neg_add | Replace Sub node with a negation (Mul by -1) -> Add. May be impacted by asymmetric quantization | False |
| 35 | convert_conv_even_filter_to_odd | Replaces even-sized convolutions with the next-size-up Odd filer as a workaround for unimplemented even-sized kernels. This is supported up to 6x6 (replace w/ 7x7). This rule will insert an additional (asymmetric) Pad before, and TIDL import will likely add a corresponding Crop layer during parsing. | False |
| 36 | remove_duplicates | Removes duplicate nodes, i.e. those that take the same inputs and have the same parameters. These should produce identical results and can be skipped. | False |
| 37 | remove_unity_resize | Remove "Resize" notes with unity scaling factor (scales=1) | False |
| 38 | insert_1x1_conv_before_depthtospace | Add a 1x1 conv before depthtospace operation as this layer fuses into the previous conv | False |
| 39 | convert_depth2space_to_reshp_tr_reshp | Replace the DepthToSpace operation with reshape->transpose->reshape operation | True |
| 40 | convert_space2depth_to_reshp_tr_reshp |  Replace the SpaceToDepth operation with reshape->transpose->reshape operation | True |
| 41 | convert_tanhgelu_to_erfgelu | Replace the gelu based on tanh to the originial erf based gelu  | True |
| 42 | support_broadcast_ops_constant_input | Replaces the constants in elt-wise arithmetic operators to prevent multidimensional broadcast or cross-broadcast | False |

### NOTE
1. This module performs some optimizations on the model and one of the optimization is in early stage named as "split_batch_dim_to_parallel_input_branches". This optimization changes a network with its partial structure with batch to multiple parallel branches in order to have TIDL-RT compatible structure. As of now the "batch specific optimization" is **experimental and at early stage** and require user to provide the start and end node names where the batch dimension needs to be replaced with parallel branches. (*Check batch.py for these two global variables named START_NODE_NAME and END_NODE_NAME*) In future support will be added to automatically detect these nodes and these variables will be removed.
2. Two optimization rules provided in (RGB_YUV_model_converter.py and onnx_model_opt.py) placed  at one directory above shall be combined with this module in future, but currently can be continued to be used as independent optimization scripts
