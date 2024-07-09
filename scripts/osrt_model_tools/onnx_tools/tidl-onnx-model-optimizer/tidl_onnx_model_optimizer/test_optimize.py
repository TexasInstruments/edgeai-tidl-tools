from tidl_onnx_model_optimizer import optimize
from tidl_onnx_model_optimizer.ops import test_optimizers

model_name = ""

optimizers = test_optimizers()
optimizers = None
optimize(model_name, custom_optimizers=optimizers)
