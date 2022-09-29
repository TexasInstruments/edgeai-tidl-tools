## This script contains some examples of creating unit layer level test cases for ONNX models

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
import numpy as np


input1 = torch.randn(1, 3, 128, 128)
input2 = torch.randn(1, 3, 128, 128)
input3 = torch.randn(1, 3, 1, 1)
input4 = torch.randn(11, 1, 128)

models_path = "../unit_test_models/"
"""
#------------------------------------------------------------------------------------------
class AvgPoolTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size = 3, stride = 1)
    
    def forward(self,x):
        x = self.pool(x)
        return x

#Pool test
if __name__ == "__main__":
    model = AvgPoolTest()
    torch.onnx.export(model, input1, models_path + "avg_pool_2x2_s1.onnx", verbose = False, opset_version=11, input_names=['input'], output_names=['output'])
    onnx_model = onnx.load(models_path + "avg_pool_2x2_s1.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "avg_pool_2x2_s1.onnx")
"""
#------------------------------------------------------------------------------------------
class AddEltTest(nn.Module):
    def forward(self,x, y):
        y = torch.add(x, y)
        return y

#Add elt wise test
if __name__ == "__main__":
    model = AddEltTest()
    torch.onnx.export(model, (input1, input2), models_path + "add_eltwise.onnx", verbose = False, opset_version=11, input_names=['input1', 'input2'], output_names=['output'])
    onnx_model = onnx.load(models_path + "add_eltwise.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "add_eltwise.onnx")
"""
#------------------------------------------------------------------------------------------
class AddNonEltTest(nn.Module):
    def forward(self,x, y):
        z = torch.add(x, y)
        return z
if __name__ == "__main__":
    model = AddNonEltTest()
    torch.onnx.export(model, (input1, input3), models_path + "add_noneltwise.onnx", verbose = False, opset_version=11, input_names=['input1', 'input2'], output_names=['output'])
    onnx_model = onnx.load(models_path + "add_noneltwise.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "add_noneltwise.onnx")

#------------------------------------------------------------------------------------------
class AddConstTest(nn.Module):
    def forward(self,x):
        y = x + 2
        return y

#Add constant test
if __name__ == "__main__":
    model = AddConstTest()
    torch.onnx.export(model, input1, models_path + "add_const.onnx", verbose = False, opset_version=11, input_names=['input'], output_names=['output'])
    onnx_model = onnx.load(models_path + "add_const.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "add_const.onnx")

#------------------------------------------------------------------------------------------
class AddConstVecTest(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        const = torch.tensor(np.random.rand(3,1,1), dtype = torch.float32)
        y = x + const
        return y

#Add constant vector test
if __name__ == "__main__":
    model = AddConstVecTest()
    torch.onnx.export(model, input1, models_path + "add_const_vec.onnx", verbose = False, opset_version=11, input_names=['input'], output_names=['output'])
    onnx_model = onnx.load(models_path + "add_const_vec.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "add_const_vec.onnx")

#------------------------------------------------------------------------------------------
class AddEltTest(nn.Module):
    def forward(self,x, y):
        y = torch.add(x, y)
        return y

#Add elt wise test
if __name__ == "__main__":
    model = AddEltTest()
    torch.onnx.export(model, (input1, input2), models_path + "add_eltwise.onnx", verbose = False, opset_version=11, input_names=['input1', 'input2'], output_names=['output'])
    onnx_model = onnx.load(models_path + "add_eltwise.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "add_eltwise.onnx")

#------------------------------------------------------------------------------------------
class AddNonEltTest(nn.Module):
    def forward(self,x, y):
        z = torch.add(x, y)
        return z
if __name__ == "__main__":
    model = AddNonEltTest()
    torch.onnx.export(model, (input1, input3), models_path + "add_noneltwise.onnx", verbose = False, opset_version=11, input_names=['input1', 'input2'], output_names=['output'])
    onnx_model = onnx.load(models_path + "add_noneltwise.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "add_noneltwise.onnx")

#------------------------------------------------------------------------------------------
#----------------------------   MULTIPLICATION TEST CASES  --------------------------------------------------------------
class MulConstTest(nn.Module):
    def forward(self,x):
        y = x * 2
        return y

#Add constant test
if __name__ == "__main__":
    model = MulConstTest()
    torch.onnx.export(model, input1, models_path + "mul_const.onnx", verbose = False, opset_version=11, input_names=['input'], output_names=['output'])
    onnx_model = onnx.load(models_path + "mul_const.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "mul_const.onnx")

#------------------------------------------------------------------------------------------
class MulConstVecTest(nn.Module):
    def forward(self,x):
        const = torch.tensor(np.random.rand(3,1,1), dtype = torch.float32)
        y = x * const
        return y

#Add constant vector test
if __name__ == "__main__":
    model = MulConstVecTest()
    torch.onnx.export(model, input1, models_path + "mul_const_vec.onnx", verbose = False, opset_version=11, input_names=['input'], output_names=['output'])
    onnx_model = onnx.load(models_path + "mul_const_vec.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "mul_const_vec.onnx")


#------------------------------------------------------------------------------------------
class MulEltTest(nn.Module):
    def forward(self,x, y):
        y = torch.mul(x, y)
        return y

#Mul elt wise test
if __name__ == "__main__":
    model = MulEltTest()
    torch.onnx.export(model, (input1, input2), models_path + "mul_eltwise.onnx", verbose = False, opset_version=11, input_names=['input1', 'input2'], output_names=['output'])
    onnx_model = onnx.load(models_path + "mul_eltwise.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "mul_eltwise.onnx")

#------------------------------------------------------------------------------------------
class MulNonEltTest(nn.Module):
    def forward(self,x, y):
        z = torch.mul(x, y)
        return z
if __name__ == "__main__":
    model = MulNonEltTest()
    torch.onnx.export(model, (input1, input3), models_path + "mul_noneltwise.onnx", verbose = False, opset_version=11, input_names=['input1', 'input2'], output_names=['output'])
    onnx_model = onnx.load(models_path + "mul_noneltwise.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "mul_noneltwise.onnx")

#----------------------------------- Inner product test case ------------------------------
class InnerProductTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.gemm = nn.Linear(128, 256)
    def forward(self,x):
        z = self.gemm(x)
        return z
if __name__ == "__main__":
    model = InnerProductTest()
    input = torch.randn(1,128)
    torch.onnx.export(model, input, models_path + "inner_product.onnx", verbose = False, opset_version=11, input_names=['input'], output_names=['output'])
    onnx_model = onnx.load(models_path + "inner_product.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "inner_product.onnx")


#------------------------------------------------------------------------------------------
class InnerProductNonflattenTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.gemm = nn.Linear(128, 256)
    def forward(self,x):
        z = self.gemm(x)
        return z
if __name__ == "__main__":
    model = InnerProductNonflattenTest()
    input = torch.randn(3,128)
    torch.onnx.export(model, input, models_path + "inner_product_nonflatten.onnx", verbose = False, opset_version=11, input_names=['input'], output_names=['output'])
    onnx_model = onnx.load(models_path + "inner_product_nonflatten.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "inner_product_nonflatten.onnx")

#----------------------------------- Inner product test case ------------------------------
class InnerProductTest_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.gemm = nn.Linear(128, 256)
    def forward(self,x):
        z = self.gemm(x)
        return z
if __name__ == "__main__":
    model = InnerProductTest_1()
    input = torch.randn(1,1,1,128)
    torch.onnx.export(model, input, models_path + "inner_product_1.onnx", verbose = False, opset_version=11, input_names=['input'], output_names=['output'])
    onnx_model = onnx.load(models_path + "inner_product_1.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "inner_product_1.onnx")

#------------------------------------------- Reshape test cases ---------------------------
class ReshapeTest(nn.Module):
    def forward(self,x):
        z = torch.reshape(x, (-1,))
        return z
if __name__ == "__main__":
    model = ReshapeTest()
    input = torch.randn(1,1,1,128)
    torch.onnx.export(model, input, models_path + "reshape.onnx", verbose = False, opset_version=11, input_names=['input'], output_names=['output'])
    onnx_model = onnx.load(models_path + "reshape.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "reshape.onnx")

#----------------------------------------- Conv pad test case ----------------------------
class ConvTest(torch.nn.Module):
    def forward(self, x):
        conv = torch.nn.Conv2d(3,16,kernel_size=3,padding="SAME_LOWER")
        y = conv(x)        
        return y
if __name__ == "__main__":
    model = ConvTest()
    input = torch.randn(1,3,10,10)
    torch.onnx.export(model, input, models_path + "conv_sameLower_base.onnx", verbose = False, opset_version=11, input_names=['input'], output_names=['output'])
    onnx_model = onnx.load(models_path + "conv_sameLower_base.onnx")
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, models_path + "conv_sameLower_base.onnx")
"""
