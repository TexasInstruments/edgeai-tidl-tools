from tidl_onnx_model_optimizer import optimize
from tidl_onnx_model_optimizer.ops import test_optimizers

model_name = "" # add the path to your onnx file here

optimizers = test_optimizers() # need to modify this to debug your transformation
optimize(model_name, custom_optimizers=optimizers)
