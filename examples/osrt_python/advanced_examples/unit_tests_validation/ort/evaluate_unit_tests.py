## This script can be used to compare the ARM vs TIDL 32 bit floating point outputs for ONNX models 
## as dumped by onnxrt_ep.py.
##
## Usage : Specify model name in the 'models' list below as specified in the onnxrt_ep.py 
## Output format : {model_name : {output_name_i : max_diff_i}}

models = ['add_eltwise']  # specify model names here

import numpy as np
import onnxruntime as rt
import os
import sys
# directory reach
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
# setting path
sys.path.append(parent)
from common_utils import *

models_base_path = '../unit_test_models'

diff_info = {}

for model_name in models:
    diff_info[model_name] = {}
    config = models_configs[model_name]
    #create session to get output names
    EP_list = ['CPUExecutionProvider']
    sess = rt.InferenceSession(config['model_path'] , providers=EP_list)
    output_details = sess.get_outputs()
    for output_detail in output_details:
        output_ref = '../outputs/output_ref/onnx/'+ os.path.basename(config['model_path']) + '_' + output_detail.name + '.bin'
        output_test = '../outputs/output_test/onnx/' +  os.path.basename(config['model_path']) + '_' + output_detail.name + '.bin'
        o1 = np.fromfile(output_ref, dtype = np.float32)
        o2 = np.fromfile(output_test, dtype = np.float32)
        diff = abs(o1 - o2)
        max_diff = max(diff)
        diff_info[model_name][output_detail.name] = max_diff

print(diff_info)