from tidl_onnx_model_optimizer import optimize
from tidl_onnx_model_optimizer.ops import test_optimizers
import numpy as np
import onnxruntime

model_name = "" # add the path to your onnx file here

optimizers = test_optimizers() # need to modify this to debug your transformation
# optimizers = None # checks the default setting
optimize(model_name, custom_optimizers=optimizers)

optimized_model_path = '/'.join(model_name.split('/')[:-1]) + f"/optimized_{model_name.split('/')[-1]}" 

# check if the output of the original and the converted onnx model matches
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
session1 = onnxruntime.InferenceSession(model_name, sess_options, providers=['CPUExecutionProvider'])
session2 = onnxruntime.InferenceSession(optimized_model_path, sess_options, providers=['CPUExecutionProvider'])
input_name1 = session1.get_inputs()[0].name
input_name2 = session2.get_inputs()[0].name
onnx_input = np.ones(session1.get_inputs()[0].shape, dtype=np.float32)
output1 = session1.run([], {input_name1: onnx_input})
output2 = session2.run([], {input_name2: onnx_input})

for i in range(len(output1)):
    print(f"Error obtained in the onnx optimization is : {(output1[i]-output2[i]).mean()}")
