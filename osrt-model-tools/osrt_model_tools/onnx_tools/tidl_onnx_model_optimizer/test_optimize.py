from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer import optimize
from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer.ops import test_optimizers, get_optimizers
import numpy as np
import onnxruntime
import os

import logging
logging.basicConfig(level=logging.INFO) 

model_name = "vit_tiny_patch16_224_simp.onnx" # add the path to your onnx file here

# optimizers = test_optimizers() # need to modify this to debug your transformation
# optimizers = get_optimizers(bucket_flags=['LAYOUT_ALL']) # need to modify this to debug bucket transformations
optimizers = get_optimizers() # need to modify this to debug all transformations
# optimizers = None # checks the default setting
directory, file = os.path.split(model_name)
optimized_model_path = os.path.join(directory, 'optimized_'+file)

optimize(model_name, out_model=optimized_model_path, custom_optimizers=optimizers, verbose=True)

# check if the output of the original and the converted onnx model matches
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
session1 = onnxruntime.InferenceSession(model_name, sess_options, providers=['CPUExecutionProvider'])
session2 = onnxruntime.InferenceSession(optimized_model_path, sess_options, providers=['CPUExecutionProvider'])

input_dict = {}
dtype_mapping = {'tensor(float)' : np.float32, 
                 'tensor(int64)' : np.int64,
                 'tensor(uint8)' : np.uint8,
                 'tensor(int32)' : np.int32}

for inp in session1.get_inputs():
    input_dict[inp.name] = np.ones(inp.shape, dtype=dtype_mapping[inp.type])
output1 = session1.run([], input_dict)
output2 = session2.run([], input_dict)

for i in range(len(output1)):
    print(f"Absolute error obtained in the onnx optimization for output {i} is : {(output1[i] - output2[i]).mean()}")
    print(f"Percentage error obtained in the onnx optimization for output {i} is : {100*abs(output1[i] - output2[i]).mean()/(output1[i].max()-output1[i].min())}%")


