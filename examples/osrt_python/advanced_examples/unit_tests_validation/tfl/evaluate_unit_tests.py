import numpy as np


#models = ['add_eltwise', 'add_const', 'add_const_vec', 'add_noneltwise']
#models = ['mul_eltwise', 'mul_const', 'mul_const_vec', 'mul_noneltwise']
models = ['fully_connected', 'fully_connected_no_bias', 'fully_connected_relu']
models = ['model_MNISTfashion_Flatten']
models = ['leakyRelu']
models = ['deconv2d_2x2_s2', 'deconv2d_3x3_s2', 'deconv2d_4x4_s2_nobias', 'deconv2d_4x4_s2_bias']
models = ['converted_model_2D_W9_keras']

diff_info = {}

for model_name in models:
    output_ref = '../outputs/output_ref/tflite/'+ model_name + '.tflite.bin'
    output_test = '../outputs/output_test/tflite/' +  model_name + '.tflite.bin'
    # output_ref = '../traces_ref/OUT20.bin'
    # output_test = '/tmp/tidl_trace0007_00001_00001_01600x00001_float.bin'
    o1 = np.fromfile(output_ref, dtype = np.float32)
    o2 = np.fromfile(output_test, dtype = np.float32)
    diff = abs(o1 - o2)
    max_diff = max(diff)
    diff_info[model_name] = max_diff
    print(max_diff)

print(diff_info)