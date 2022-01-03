# Copyright (c) {2015 - 2021} Texas Instruments Incorporated
#
# All rights reserved not granted herein.
#
# Limited License.
#
# Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
# license under copyrights and patents it now or hereafter owns or controls to make,
# have made, use, import, offer to sell and sell ("Utilize") this software subject to the
# terms herein.  With respect to the foregoing patent license, such license is granted
# solely to the extent that any such patent is necessary to Utilize the software alone.
# The patent license shall not apply to any combinations which include this software,
# other than combinations with devices manufactured by or for TI ("TI Devices").
# No hardware patent is licensed hereunder.
#
# Redistributions must preserve existing copyright notices and reproduce this license
# (including the above copyright notice and the disclaimer and (if applicable) source
# code license limitations below) in the documentation and/or other materials provided
# with the distribution
#
# Redistribution and use in binary form, without modification, are permitted provided
# that the following conditions are met:
#
# *       No reverse engineering, decompilation, or disassembly of this software is
# permitted with respect to any software provided in binary form.
#
# *       any redistribution and use are licensed by TI for use only with TI Devices.
#
# *       Nothing shall obligate TI to provide you with source code for the software
# licensed and provided to you in object code.
#
# If software source code is provided to you, modification and redistribution of the
# source code are permitted provided that the following conditions are met:
#
# *       any redistribution and use of the source code, including any resulting derivative
# works, are licensed by TI for use only with TI Devices.
#
# *       any redistribution and use of any object code compiled from the source code
# and any resulting derivative works, are licensed by TI for use only with TI Devices.
#
# Neither the name of Texas Instruments Incorporated nor the names of its suppliers
#
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# DISCLAIMER.
#
# THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

import flatbuffers
import tflite_model.Model 
import tflite_model.BuiltinOperator
import tflite_model.BuiltinOptions
import tflite_model.Tensor
import tflite_model.TensorType
import copy 
import struct

def addNewOperator(modelT, operatorBuiltinCode):
    new_op_code                       = copy.deepcopy(modelT.operatorCodes[0])
    new_op_code.deprecatedBuiltinCode = operatorBuiltinCode
    modelT.operatorCodes.append(new_op_code)
    return (len(modelT.operatorCodes) - 1)

def getArgMax_idx(modelT):
    idx = 0
    for op in modelT.operatorCodes:
        if(op.deprecatedBuiltinCode == tflite_model.BuiltinOperator.BuiltinOperator.ARG_MAX):
            break
        idx = idx + 1
    return idx

def setTensorProperties(tensor, dataType, scale, zeroPoint):
    tensor.type                   = dataType
    tensor.quantization.scale     = [scale]
    tensor.quantization.zeroPoint = [zeroPoint]

def createTensor(modelT, dataType, quantization, tensorShape, tensorName):
    newTensor              = copy.deepcopy(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]])
    newTensor.type         = dataType
    newTensor.quantization = quantization
    newTensor.shape        = tensorShape
    newTensor.name         = tensorName
    return newTensor


def tidlTfliteModelOptimize(in_model_path, out_model_path, scaleList=[0.0078125,0.0078125,0.0078125], meanList=[128.0, 128.0, 128.0]):
    #Open the tflite model
    print(in_model_path)
    meanList = [x * -1 for x in meanList]
    modelBin = open(in_model_path, 'rb').read()
    if modelBin is None:
        print(f'Error: Could not open file {in_model_path}')
        return
    modelBin = bytearray(modelBin)
    model = tflite_model.Model.Model.GetRootAsModel(modelBin, 0)
    modelT = tflite_model.Model.ModelT.InitFromObj(model)

    #Add operators needed for preprocessing:
    setTensorProperties(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]], tflite_model.TensorType.TensorType.UINT8, 1.0, 0)
    mul_idx = addNewOperator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.MUL)
    add_idx = addNewOperator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.ADD)
    cast_idx = addNewOperator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.CAST)

    in_cast_idx = addNewOperator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.CAST)
    #Find argmax in the network:
    argMax_idx = getArgMax_idx(modelT)

    #Create a tensor for the "ADD" operator:
    bias_tensor = createTensor(modelT, tflite_model.TensorType.TensorType.FLOAT32, None, [modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape[3]], bytearray(str("Preproc-bias"),'utf-8'))
    #Create a new buffer to store mean values:
    new_buffer = copy.copy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].buffer])
    new_buffer.data = struct.pack('%sf' % len(meanList), *meanList)
    modelT.buffers.append(new_buffer)
    new_buffer_idx = len(modelT.buffers) - 1
    bias_tensor.buffer = new_buffer_idx

    #Create a tensor for the "MUL" operator
    scale_tensor = copy.deepcopy(bias_tensor)
    scale_tensor.name  = bytearray(str("Preproc-scale"),'utf-8')
    #Create a new buffer to store the scale values:
    new_buffer = copy.copy(new_buffer)
    new_buffer.data = struct.pack('%sf' % len(scaleList), *scaleList)
    modelT.buffers.append(new_buffer)
    new_buffer_idx = len(modelT.buffers) - 1
    scale_tensor.buffer = new_buffer_idx

    #Append tensors into the tensor list:
    modelT.subgraphs[0].tensors.append(bias_tensor)
    bias_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 
    modelT.subgraphs[0].tensors.append(scale_tensor)
    scale_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 
    new_tensor = copy.deepcopy(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]])
    new_tensor.name = bytearray((str(new_tensor.name, 'utf-8') + str("/Mul")),'utf-8')
    modelT.subgraphs[0].tensors.append(new_tensor)
    new_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 
    new_buffer = copy.deepcopy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]].buffer])
    modelT.buffers.append(new_buffer)
    new_buffer_idx = len(modelT.buffers) - 1
    modelT.subgraphs[0].tensors[new_tensor_idx].buffer = new_buffer_idx

    #Add the MUL Operator for scales:
    new_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
    modelT.subgraphs[0].operators.insert(0,new_op)
    modelT.subgraphs[0].operators[0].outputs[0] = new_tensor_idx
    modelT.subgraphs[0].operators[0].inputs = [modelT.subgraphs[0].operators[1].inputs[0],scale_tensor_idx]
    modelT.subgraphs[0].operators[1].inputs[0] = new_tensor_idx
    modelT.subgraphs[0].tensors[new_tensor_idx].shape = modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape
    modelT.subgraphs[0].operators[0].opcodeIndex = mul_idx
    modelT.subgraphs[0].operators[0].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.MulOptions
    modelT.subgraphs[0].operators[0].builtinOptions = tflite_model.MulOptions.MulOptionsT()

    #Add the ADD operator for mean:
    new_tensor = copy.deepcopy(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]])
    new_tensor.name = bytearray((str(new_tensor.name, 'utf-8') + str("/Bias")),'utf-8')
    modelT.subgraphs[0].tensors.append(new_tensor)
    new_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 
    new_buffer = copy.deepcopy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]].buffer])
    modelT.buffers.append(new_buffer)
    new_buffer_idx = len(modelT.buffers) - 1
    modelT.subgraphs[0].tensors[new_tensor_idx].buffer = new_buffer_idx
    new_op_code = copy.deepcopy(modelT.operatorCodes[0])
    new_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
    modelT.subgraphs[0].operators.insert(0,new_op)
    modelT.subgraphs[0].operators[0].outputs[0] = new_tensor_idx
    modelT.subgraphs[0].operators[0].inputs = [modelT.subgraphs[0].operators[1].inputs[0],bias_tensor_idx]
    modelT.subgraphs[0].operators[1].inputs[0] = new_tensor_idx
    modelT.subgraphs[0].tensors[new_tensor_idx].shape = modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape
    modelT.subgraphs[0].operators[0].opcodeIndex = add_idx
    modelT.subgraphs[0].operators[0].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.AddOptions
    modelT.subgraphs[0].operators[0].builtinOptions = tflite_model.AddOptions.AddOptionsT()

    #Add the dequantize operator:
    new_tensor = copy.deepcopy(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]])
    new_tensor.name = bytearray((str(new_tensor.name, 'utf-8') + str("/InCast")),'utf-8')
    modelT.subgraphs[0].tensors.append(new_tensor)
    new_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 
    new_buffer = copy.deepcopy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]].buffer])
    modelT.buffers.append(new_buffer)
    new_buffer_idx = len(modelT.buffers) - 1
    modelT.subgraphs[0].tensors[new_tensor_idx].buffer = new_buffer_idx
    new_op_code = copy.deepcopy(modelT.operatorCodes[0])
    new_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
    modelT.subgraphs[0].operators.insert(0,new_op)
    modelT.subgraphs[0].operators[0].outputs[0] = new_tensor_idx
    modelT.subgraphs[0].operators[0].inputs = [modelT.subgraphs[0].operators[1].inputs[0]]
    modelT.subgraphs[0].operators[1].inputs[0] = new_tensor_idx
    modelT.subgraphs[0].tensors[new_tensor_idx].shape = modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape
    modelT.subgraphs[0].operators[0].opcodeIndex = in_cast_idx
    modelT.subgraphs[0].operators[0].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.CastOptions
    modelT.subgraphs[0].operators[0].builtinOptions = tflite_model.CastOptions.CastOptionsT()
    modelT.subgraphs[0].operators[0].builtinOptions.inDataType = tflite_model.TensorType.TensorType.UINT8
    modelT.subgraphs[0].tensors[new_tensor_idx].type  = tflite_model.TensorType.TensorType.FLOAT32

    #Detect and convert ArgMax's output data type:
    for operator in modelT.subgraphs[0].operators:
        #Find ARGMAX:
        if(operator.opcodeIndex == argMax_idx):
            if(modelT.subgraphs[0].tensors[operator.inputs[0]].shape[3] < 256): #Change dType only if #Classes can fit in UINT8 
                #Add CAST Op on ouput of Argmax:
                new_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
                modelT.subgraphs[0].operators.append(new_op)
                new_op_idx = len(modelT.subgraphs[0].operators) - 1
            
                modelT.subgraphs[0].operators[new_op_idx].outputs[0] = operator.outputs[0]

                new_tensor = copy.deepcopy(modelT.subgraphs[0].tensors[operator.outputs[0]])
                new_tensor.name = bytearray((str(new_tensor.name, 'utf-8') + str("_org")),'utf-8')
                modelT.subgraphs[0].tensors.append(new_tensor)
                new_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 
                new_buffer = copy.deepcopy(modelT.buffers[modelT.subgraphs[0].tensors[operator.outputs[0]].buffer])
                modelT.buffers.append(new_buffer)
                new_buffer_idx = len(modelT.buffers) - 1
                modelT.subgraphs[0].tensors[new_tensor_idx].buffer = new_buffer_idx

                operator.outputs[0] = new_tensor_idx

                modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[new_op_idx].outputs[0]].type  = tflite_model.TensorType.TensorType.UINT8

                modelT.subgraphs[0].operators[new_op_idx].inputs[0] = new_tensor_idx
                modelT.subgraphs[0].operators[new_op_idx].opcodeIndex = cast_idx
                modelT.subgraphs[0].operators[new_op_idx].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.CastOptions
                modelT.subgraphs[0].operators[new_op_idx].builtinOptions = tflite_model.CastOptions.CastOptionsT()
                modelT.subgraphs[0].operators[new_op_idx].builtinOptions.outDataType = tflite_model.TensorType.TensorType.UINT8


    # Packs the object class into another flatbuffer.
    b2 = flatbuffers.Builder(0)
    b2.Finish(modelT.Pack(b2), b"TFL3")
    modelBuf = b2.Output() 
    newFile = open(out_model_path, "wb")
    newFile.write(modelBuf)