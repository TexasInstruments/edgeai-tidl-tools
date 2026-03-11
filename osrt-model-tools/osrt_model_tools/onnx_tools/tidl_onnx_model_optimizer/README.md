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

## Bucket-Based Optimization

The optimizer supports a **bucket** mechanism to group related optimizations and control their execution order. Buckets are logical collections of optimization passes that can be enabled or disabled as a group. This allows users to easily apply sets of related transformations, ensuring dependencies and execution order are respected.
> **Note:** Buckets are for better categorization and creating structure. All optimizations will work even without using buckets; you can enable and use individual optimizations directly.


### How Buckets Work

- Each bucket (e.g., `BASIC_ALL`, `EXTENDED_ALL`, `LAYOUT_ALL`) contains a list of optimization keys.
- Buckets can have dependencies on other buckets, enforced via a dependency graph.
- When a bucket is enabled, all optimizations within it are considered for execution, and any dependent buckets are automatically enabled.
- The optimizer determines whether to run in "bucket mode" or "individual mode" based on the provided flags.

### Usage

To enable a bucket, use the `get_optimizers` function with the `bucket_flags` argument:

```python
from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer.ops import get_optimizers

optimizers = get_optimizers(bucket_flags=['LAYOUT_ALL'])
```

This will enable all optimizations in the `LAYOUT_ALL` bucket and its dependencies.

You can also combine bucket flags with individual optimization flags for fine-grained control.

### Example Buckets

- `BASIC_ALL`: Foundational optimizations (e.g., removing duplicates, basic shape changes).
- `EXTENDED_ALL`: More advanced or dependent optimizations, depends on `BASIC_ALL`.
- `LAYOUT_ALL`: Layout-related optimizations, depends on `EXTENDED_ALL`.

See [ops.py](ops.py) for the full list of buckets and their contents.

## Operations
The different optimizations performed are summarized here along with their default flag value (Enabled = True, Disabled = False).

| Sl. no. | Function (flag)                       | Summary                               |       Default     |
|:------: | :------------------------------------ |:------------------------------------: | :---------------- |
| 1 | add_input_normalization| Add input normalization for uint8 inputs to the model, needs a dictiorary containing "input_mean" and "input_scale", <br> - if none of the two are provided, the normalization is skipped <br> - if only one is provided, the other is set to default values | False |
| 2 | convert_resize_params_size_to_scale | Resize operator can specify either size of scale parameter in input, but TIDL does not support size input params. This function converts size to corresposding scale. For e.g, with input [3, 256, 256] and size input [3, 128, 128], it will convert to scales [1, 2, 2]| False |
| 3 | convert_concat_axis_width_to_channel | TIDL only supports concat on channel axis. This function converts Concat layer with width axis to Concat layer with channel axis adjusting the input and output accordingly with Reshapes | False |
| 4 | convert_maxpool_to_cascaded_maxpool | Convert large MaxPool kernels (>3x3) to cascaded 3x3 and 2x2 layers with exact receptive field matching. Supports odd kernels (5x5, 7x7, ...) with stride 1 or 2, and even kernels (4x4, 6x6, ...) with stride 2 only. | True |
| 5 | expand_multiaxes_reducemean_to_single_axis_reducemeans | The ReduceSum layer with multi-axis is replaced with cascaded multiple layers, e.g., "Reshape + ReduceSum + ReduceSum + ... + Reshape (if keepdims=1)". Contiguous axes are merged via Reshape, then each merged dimension is reduced with single-axis ReduceSum operations. | True |
| 6 | convert_gemm_to_matmul_and_add | Gemm layer with constant B input in converted to Matmul and Gemm bias (if exists) is converted to a following Add layer | False |
| 7 | convert_matmul_to_conv_1x1s1 | Function to convert MatMul layer to Convolution with kernel 1x1, stride 1x1. Only works for MatMuls with input dimensions not equal to 3 (i.e., 2 or >= 4 works) | False |
| 8 | convert_large_global_avg_pooling_to_matmul | Global average pooling with large HxW values might be unoptimal, converting the input with a reshape from HxW to 1xHW and doing MatMul with a const tensor of dim HWx1 and value of 1/HW | True |
| 9 | convert_gather_scalar_to_1d | Replace scalar Gather indices with 1D arrays and add Reshape to maintain output shape. | True |
| 10 | convert_batchnorm_input_to_4D |  Batchnorm input with less than 4 dimension is converted to 4 dimension by adding 1's at the end, done using Reshaped before and after the layer. TIDL supports only 4D batchnorm (NCHW) with batchnorm on the channel | True |
| 11 | attention_block_optimization | Attention block optimization function, identifies attention blocks and performs TIDL specific optimizations on the attention blocks as a whole | False |
| 12 | split_batch_dim_to_parallel_input_branches | If network has batch dimensions to some layers which does not suppport batch dim in TIDL framework, duplicate the layer and split in multiple branches so as each batch gets treated as different input to different branch | False |
| 13 | convert_softmax_axis_height_to_width | The SoftMax layer with operation in the height dimension is replaced with Transpose -> SoftMax -> Transpose to satisfy constraint of SoftMax layer only occuring in width dimension | False |
| 14 | convert_softmax_unsupported_axis_to_width | TIDL hardware supports Softmax operations only on width (axis=-1) and height (axis=-2) dimensions. The Softmax layer with operations on any other axis is replaced with Transpose → Softmax → Transpose pattern to convert unsupported axes to width dimension (axis=-1). | True |
| 15 | push_large_channel_dim_to_height_for_width_wise_softmax | When a softmax has high value of dimensions channel and upper it performs unoptimal. But reshaping the shape to have a larger height can make it more efficient. Hence Softmax is changed to Reshape -> Softmax -> Reshape | True |
| 16 | convert_conv_large_pad_to_smaller_kernel | Convolution layer with large kernels and small inputs might be unsupported when pad is greater than the input dimension. This can be converted to Conv with smaller kernel and less pad for support | True |
| 17 | expand_layernorm_to_component_ops | The LayerNormalization-17 layer from ONNX is not supported by TIDL. We can expand this layer to it's fundamental operators to make it supported in TIDL | False |
| 18 | push_matmul_channel_in_height | Matmul layers with one input broadcasted across channel and other input with small plane size can have the channel and height axis merged to get optimized performance | False |
| 19 | expand_slice_across_multiple_axis | Slice along a single axis is currently supported for TIDL import. This will split the slice into multiple slices each acting on a single axis. | True |
| 20 | convert_instancenorm_to_layernorm | InstanceNormalisation is not supported in TIDL, converting it to LayerNorm with the same functionality. | False |
| 21 | convert_unsqueeze_to_reshape | Converts the Unsqueeze layer to reshape layer for support. | False |
| 22 | add_bias_qdq | Adds the bias quantization to conv layers if not already there (Weight_params * Act_params) | False |
| 23 | remove_quantize_initializer | Removes the Quantization node in initialisers (reduces the model size as input becomes 8-bit) - Use only for PT2E exported models (quantization=3) | True |
| 24 | remove_duplicate_quantize_dequantize | Removes the duplicate sequential Q-DQ layers (keeps the first quant params) | False |
| 25 | convert_neg_to_mul | Converts the Neg layer (from RoPE) to mul by -1 | True |
| 26 | convert_expand_to_reshape_and_concat | Converts the expand layer to reshape and concat | False |
| 27 | convert_single_concat_to_consecutive_concats | Convert a concat which works as expanding a dimension of a tensor (1x1x10 -> 1x5x10) to multiple consecutive concats which only takes 2 inputs at once, thus in the example, we would have 4 different concats.  | True |
| 28 | convert_conv_7x7_stride4_to_stride1 | Few models(segformer) has a convolution layer with 7x7 kernel and 4 stride, converting the layer to the one with a stride of 1 using combination of maxpool and conv  | True |
| 29  | convert_2_dimension_slice_to_maxpool | Slice if present in 2 axes, with same steps, it is converted to a corresponding maxpool with kernel size of 1, transpose also are inserted if channel not in 2nd dimension | False |
| 30 | convert_unsupported_argmax_to_supported | Converts ArgMax nodes to TIDL-compatible format by ensuring keepdims=1, moving axis to -3 position (for 3D/4D), and handling select_last_index via data reversal. | True |
| 31 | hf_attention_block_optimization | Attention block optimization function, identifies attention blocks and performs TIDL specific optimizations on the attention blocks as a whole | True |
| 32 | expand_multiaxes_reducesum_to_single_axis_reducesums | The ReduceSum layer with multi-axis is replaced with cascaded multiple layers, e.g., "Reshape + ReduceSum + ReduceSum + ... + Reshape (if keepdims=1)". Contiguous axes are merged via Reshape, then each merged dimension is reduced with single-axis ReduceSum operations. | True |
| 33 | convert_resize_params_size_to_scale_dynamic_batch | Finds Resize nodes that use a sequence of nodes that dynamically determine output sizes, which are added during export. The rule determines the static 'scales', and removes the dynamic nodes such that the Reize node is supported | False |
| 34 | replace_mean_with_eltwise | Replaced Mean of 2 tensors with Add + Multiply by 0.5 (since Div is also not supported). >2 inputs is not supported, but could be implemented without much difficulty | False |
| 35 | replace_sub_with_neg_add | Replace Sub node with a negation (Mul by -1) -> Add. May be impacted by asymmetric quantization | False |
| 36 | convert_conv_even_filter_to_odd | Replaces even-sized convolutions with the next-size-up Odd filer as a workaround for unimplemented even-sized kernels. This is supported up to 6x6 (replace w/ 7x7). This rule will insert an additional (asymmetric) Pad before, and TIDL import will likely add a corresponding Crop layer during parsing. | False |
| 37 | remove_duplicates | Removes duplicate nodes, i.e. those that take the same inputs and have the same parameters. These should produce identical results and can be skipped. | False |
| 38 | remove_unity_resize | Remove "Resize" notes with unity scaling factor (scales=1) | False |
| 39 | insert_1x1_conv_before_depthtospace | Add a 1x1 conv before depthtospace operation as this layer fuses into the previous conv | False |
| 40 | convert_depth2space_to_reshp_tr_reshp | Replace the DepthToSpace operation with reshape->transpose->reshape operation (Currently disabled as TIDL natively supports DepthToSpace when input is 4D and channels are divisible by block_size². Enable this optimization only for non-standard configurations or unsupported cases.) | False | 
| 41 | convert_space2depth_to_reshp_tr_reshp |  Replace the SpaceToDepth operation with reshape->transpose->reshape operation | True |
| 42 | convert_tanhgelu_to_erfgelu | Replace the gelu based on tanh to the originial erf based gelu  | True |
| 43 | support_broadcast_ops_constant_input | Replaces the constants in elt-wise arithmetic operators to prevent multidimensional broadcast or cross-broadcast | False |
| 44 | remove_where_layer | Remove the where layer when the condition is all True or all False | True |  
| 45 | eliminate_noop_slice | Removes Slice nodes that do not change the input tensor | True |
| 46 | eliminate_unsqueeze | Removes Unsqueeze nodes that do not change the input tensor | True |
| 47 | break_gelu_to_components | Breaks the GELU activation into its primitive operations using the erf-based formula | True |
| 48 | convert_tr_conv_stride_n_tr_to_matmul | Models having transpose -> conv(stride n) -> transpose need to be converted to reshape -> transpose -> reshape -> matmul -> add -> reshape | True | 
| 49 | optimize_reshp_tr_reshp | Optimize the reshape transpose reshape layers such that if transpose exists in consecutive axis, then it can be clubbed together such that the number of dimension are reduced. | True | 
| 50 | hf_detr_attention_block_optimization | Attention block optimization function for Hugging Face DETR, identifies attention blocks and performs TIDL specific optimizations on the attention blocks as a whole | True |
| 51 | convert_reducemax_for_height_axis | Converts ReduceMax operations to TIDL-compatible format by transforming arbitrary axis reductions into height-axis (rank = -2 position) reductions.  | True |
| 52 | replace_tile_gatherelements_with_reshape_gather | Replace Tile+GatherElements patterns with Reshape+Gather operations for better performance optimization  | True |
| 53 | replace_einsum_with_matmul_and_basic_ops | Replaces Einsum operations with equation 'bnc,bchw->bnhw' with a simplified combination of Reshape, Transpose, and MatMul operations.  | True |
| 54 | convert_matmul_with_1d_weight_to_2d_weight_and_reshape | Converts MatMul operations with 1D weight constants to use 2D weights with appropriate reshaping.  | True |
| 55 | convert_global_pooling_to_reduce_ops | Convert MaxPool/AveragePool with kernel==stride==input_size to ReduceMax/ReduceMean  | True |
| 56 | convert_tile_to_expand_for_size1_dims | Convert Tile to Expand only when repeating size-1 dimensions. | True|
| 57 | replace_expand_gatherelements_with_reshape_gather | Replace Expand+GatherElements patterns with Reshape+Gather operations for better performance optimization  | True |
| 58 | convert_non_singular_strided_slice_to_gather | Convert Non singular strided Slice to Gather operation. | True | 
| 59 | convert_patch_merging_to_reshp_tr_reshp | Converts patch merging operations to reshape-transpose-reshape sequence for better performance. | True |
| 60 | convert_reducel2_to_mul_reducesum_sqrt | Converts ReduceL2 operations to a sequence of Mul, ReduceSum, and Sqrt operations for improved compatibility. | True |
| 61 | adjust_clip_minval_maxval | Adjusts Clip operation's min and max values to ensure compatibility with TIDL. | True |
| 62 | convert_pad_above_height_axis_to_height_axis | Converts padding operations above height axis to operations on the height axis directly. | True |
| 63 | convert_single_axis_gethernd_to_gather | Converts single-axis GatherND operations to regular Gather operations for better compatibility. | True |
| 64 | break_transpose_of_width_to_dim1_dim2_of_input_more_than_4d | Breaks down transpose operations of width to dim1/dim2 when input has more than 4 dimensions. | True |
<!-- TODO add for the rest conversion rules-->

### NOTE
1. This module performs some optimizations on the model and one of the optimization is in early stage named as "split_batch_dim_to_parallel_input_branches". This optimization changes a network with its partial structure with batch to multiple parallel branches in order to have TIDL-RT compatible structure. As of now the "batch specific optimization" is **experimental and at early stage** and require user to provide the start and end node names where the batch dimension needs to be replaced with parallel branches. (*Check batch.py for these two global variables named START_NODE_NAME and END_NODE_NAME*) In future support will be added to automatically detect these nodes and these variables will be removed.
2. Two optimization rules provided in (RGB_YUV_model_converter.py and onnx_model_opt.py) placed  at one directory above shall be combined with this module in future, but currently can be continued to be used as independent optimization scripts
