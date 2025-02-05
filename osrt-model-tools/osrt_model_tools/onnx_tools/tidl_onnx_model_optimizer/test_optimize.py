from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer import optimize
from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer.ops import test_optimizers, get_optimizers
import numpy as np
import onnxruntime

# model_name = "" # add the path to your onnx file here

# optimizers = test_optimizers() # need to modify this to debug your transformation
optimizers = get_optimizers() # need to modify this to debug all transformations
# optimizers = None # checks the default setting
optimized_model_path = '/'.join(model_name.split('/')[:-1]) + f"/optimized_{model_name.split('/')[-1]}" 

optimize(model_name, out_model=optimized_model_path, custom_optimizers=optimizers, verbose=False)

# check if the output of the original and the converted onnx model matches
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
session1 = onnxruntime.InferenceSession(model_name, sess_options, providers=['CPUExecutionProvider'])
session2 = onnxruntime.InferenceSession(optimized_model_path, sess_options, providers=['CPUExecutionProvider'])

input_dict = {}
for inp in session1.get_inputs():
    input_dict[inp.name] = np.ones(inp.shape, dtype=np.float32)
output1 = session1.run([], input_dict)
output2 = session2.run([], input_dict)

for i in range(len(output1)):
    print(f"Error obtained in the onnx optimization for output {i} is : {(output1[i] - output2[i]).mean()}")

