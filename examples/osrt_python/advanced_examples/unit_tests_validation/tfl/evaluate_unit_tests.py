## This script can be used to compare the ARM vs TIDL 32 bit floating point outputs for Tflite models 
## as dumped by tflrt_delegate.py.
##
## Usage : Specify model name in the 'models' list below as specified in the tflrt_delegate.py 
## Output format : {model_name : {output_index_i : max_diff_i}}

models = ['add_const']  # specify model names here

import numpy as np
import tflite_runtime.interpreter as tflite
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
    # create interpreter to get output indices
    interpreter = tflite.Interpreter(model_path=config['model_path'], num_threads=2)
    output_details = interpreter.get_output_details()
    for output_detail in output_details:
        output_ref = '../outputs/output_ref/tflite/'+ os.path.basename(config['model_path']) + '_' + str(output_detail['index']) + '.bin'
        output_test = '../outputs/output_test/tflite/' +  os.path.basename(config['model_path']) + '_' + str(output_detail['index']) + '.bin'
        o1 = np.fromfile(output_ref, dtype = np.float32)
        o2 = np.fromfile(output_test, dtype = np.float32)
        diff = abs(o1 - o2)
        max_diff = max(diff)
        diff_info[model_name][str(output_detail['index'])] = max_diff

print(diff_info)