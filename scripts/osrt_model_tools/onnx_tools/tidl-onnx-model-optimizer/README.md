# tidl-onnx-model-optimizer
This module contains various modifications and optimization on the original ONNX graph and generates a new optimized ONNX model with the same functionality as the original model but with a more optimized structure for TIDL-RT inference.

## Setup
For setting up execute the command

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
| 1 | convert_resize_params_size_to_scale | Resize operator can specify either size of scale parameter in input, but TIDL does not support size input params. This function converts size to corresposding scale. For e.g, with input [3, 256, 256] and size input [3, 128, 128], it will convert to scales [1, 2, 2]| True |
| 2 | convert_concat_axis_width_to_channel | TIDL only supports concat on channel axis. This function converts Concat layer with width axis to Concat layer with channel axis adjusting the input and output accordingly with Reshapes | False |
| 3 | convert_maxpool_to_cascaded_maxpool | The MaxPool layer with large kernel (> 3x3) is replaced with cascaded MaxPool layers wiht 3x3 kernel. Assume that the kernel size is NxN where N is odd | False |
| 4 | convert_reducemean_to_matmul | The ReduceMean layer is replaced with the cascaded multiple layers, e.g., "Reshape + MatMul + Reshape". The attribute, "axes" of ReduceMean should be W and H dimension. ReduceMean in channel dimension is not supported | False |
| 5 | convert_gemm_to_matmul_and_add | Gemm layer with constant B input in converted to Matmul and Gemm bias (if exists) is converted to a following Add layer | True |
| 6 | convert_matmul_to_conv_1x1s1 | Function to convert MatMul layer to Convolution with kernel 1x1, stride 1x1. Only works for MatMuls with input dimensions not equal to 3 (i.e., 2 or >= 4 works) | True |
| 7 | convert_large_global_avg_pooling_to_matmul | Global average pooling with large HxW values might be unoptimal, converting the input with a reshape from HxW to 1xHW and doing MatMul with a const tensor of dim HWx1 and value of 1/HW | True |
| 8 | convert_gather_with_single_index_to_slice | Gather layer with a single index = t, can be converted to Slice [t, t+1] on the same axis | True |
| 9 | convert_batchnorm_input_to_4D |  Batchnorm input with less than 4 dimension is converted to 4 dimension by adding 1's at the end, done using Reshaped before and after the layer. TIDL supports only 4D batchnorm (NCHW) with batchnorm on the channel | True |
| 10 | attention_block_optimization | Attention block optimization function, identifies attention blocks and performs TIDL specific optimizations on the attention blocks as a whole | False |
| 11 | split_batch_dim_to_parallel_input_branches | If network has batch dimensions to some layers which does not suppport batch dim in TIDL framework, duplicate the layer and split in multiple branches so as each batch gets treated as different input to different branch | False |
| 12 | convert_softmax_axis_height_to_width | The SoftMax layer with operation in the height dimension is replaced with Transpose -> SoftMax -> Transpose to satisfy constraint of SoftMax layer only occuring in width dimension | False |
| 13 | convert_softmax_axis_channel_to_width | The SoftMax layer with operation in the channel dimension is replaced with Transpose -> SoftMax -> Transpose to satisfy constraint of SoftMax layer only occuring in width dimension | False |
| 14 | push_large_channel_dim_to_height_for_width_wise_softmax | When a softmax has high value of dimensions channel and upper it performs unoptimal. But reshaping the shape to have a larger height can make it more efficient. Hence Softmax is changed to Reshape -> Softmax -> Reshape | True |
| 15 | convert_conv_large_pad_to_smaller_kernel | Convolution layer with large kernels and small inputs might be unsupported when pad is greater than the input dimension. This can be converted to Conv with smaller kernel and less pad for support | False |
| 16 | expand_layernorm_to_component_ops | The LayerNormalization-17 layer from ONNX is not supported by TIDL. We can expand this layer to it's fundamental operators to make it supported in TIDL | True |



### NOTE
1. This module performs some optimizations on the model and one of the optimization is in early stage named as "split_batch_dim_to_parallel_input_branches". This optimization changes a network with its partial structure with batch to multiple parallel branches in order to have TIDL-RT compatible structure. As of now the "batch specific optimization" is **experimental and at early stage** and require user to provide the start and end node names where the batch dimension needs to be replaced with parallel branches. (*Check batch.py for these two global variables named START_NODE_NAME and END_NODE_NAME*) In future support will be added to automatically detect these nodes and these variables will be removed.
2. Two optimization rules provided in (RGB_YUV_model_converter.py and onnx_model_opt.py) placed  at one directory above shall be combined with this module in future, but currently can be continued to be used as independent optimization scripts