import numpy as np


#models = ['add_eltwise', 'add_const', 'add_const_vec', 'add_noneltwise']
models = ['mul_eltwise', 'mul_const', 'mul_const_vec', 'mul_noneltwise']
models = ['inner_product']
models = ['split']
models = ['ti_test_stride']
models = ['conv_k3s2_valid']
models = ['resnet18_opset9']

diff_info = {}

for model_name in models:
    output_ref = '../outputs/output_ref/onnx/'+ model_name + '.onnx.bin'
    output_test = '../outputs/output_test/onnx/' +  model_name + '.onnx.bin'
    o1 = np.fromfile(output_ref, dtype = np.float32)
    o2 = np.fromfile(output_test, dtype = np.float32)
    diff = abs(o1 - o2)
    max_diff = max(diff)
    diff_info[model_name] = max_diff
    print(max_diff)

print(diff_info)