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
import sys
import os.path
import flatbuffers
import copy
import struct
import getopt

# add local path temporarily for the import of tflite_model to work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tflite_model.Model
import tflite_model.BuiltinOperator
import tflite_model.BuiltinOptions
import tflite_model.Tensor
import tflite_model.TensorType
import numpy as np
sys.path.pop(0)


def addNewOperator(modelT, operatorBuiltinCode):
    new_op_code                       = copy.deepcopy(modelT.operatorCodes[0])
    new_op_code.deprecatedBuiltinCode = operatorBuiltinCode
    new_op_code.builtinCode = operatorBuiltinCode
    modelT.operatorCodes.append(new_op_code)
    return (len(modelT.operatorCodes) - 1)

def getArgMax_idx(modelT):
    idx = 0
    for op in modelT.operatorCodes:
        if(op.deprecatedBuiltinCode == tflite_model.BuiltinOperator.BuiltinOperator.ARG_MAX):
            break
        idx = idx + 1
    return idx

def createTensor(modelT, dataType, quantization, tensorShape, tensorName):
    newTensor              = copy.deepcopy(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]])
    newTensor.type         = dataType
    newTensor.quantization = quantization
    newTensor.shape        = tensorShape
    newTensor.name         = tensorName
    return newTensor

def removeInput(curr_inputs,val_to_remove):
        inputs = []
        for i in range(len(curr_inputs)):
            inputs.append(curr_inputs[i])
        inputs.remove(val_to_remove)
        return inputs
def appendInput(curr_inputs, new_input):
        inputs = []
        for i in range(len(curr_inputs)):
            inputs.append(curr_inputs[i])
        inputs.append(new_input)
        return inputs
    
def getWightsAndBiasData(): 
  weights = [1.164, 0.0, 1.596,
             1.164, -0.391, -0.813,
             1.164, 2.018, 0.0]
  bias= [-222.912,135.488,-276.928]
  return weights, bias

###########Function description#############
# This Function takes a RGB trained model and update the inputs to the model to accept 
# YUV(NV12) image format
# Expected NV12 format is as follows for a 224x224 RGB 
# input1 = 224x224 Y data in uint8 format
# input2 = 112x224 UV interleaved data in uint8 format
###########Function description#############
def addYUVConv(in_model_path, out_model_path):
    modelBin = open(in_model_path, 'rb').read()
    if modelBin is None:
        print(f'Error: Could not open file {in_model_path}')
        return
    modelBin = bytearray(modelBin)
    model = tflite_model.Model.Model.GetRootAsModel(modelBin, 0)
    modelT = tflite_model.Model.ModelT.InitFromObj(model)
    #Add operators needed for preprocessing:

    shape_cp = []
    for i in range(len(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape)):
        shape_cp.append(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape[i])

    modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape[3] = 1
    modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].type = tflite_model.TensorType.TensorType.FLOAT32
 
    #Create a tensor for the "UV" input:
    UV_tensor = createTensor(modelT, tflite_model.TensorType.TensorType.FLOAT32, None, [shape_cp[0], np.int32(shape_cp[1]/2), np.int32(shape_cp[2]/2), 2], bytearray(str("InputUV-semi-planar"),'utf-8'))
    UV_buffer = copy.copy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].buffer])    
    modelT.buffers.append(UV_buffer)
    UV_buffer_idx = len(modelT.buffers) - 1
    UV_tensor.buffer = UV_buffer_idx
    modelT.subgraphs[0].tensors.append(UV_tensor)
    UV_tensor_idx = len(modelT.subgraphs[0].tensors) - 1      
    modelT.subgraphs[0].inputs = appendInput(modelT.subgraphs[0].inputs,UV_tensor_idx)    
    modelT.subgraphs[0].operators[0].inputs = appendInput(modelT.subgraphs[0].operators[0].inputs,UV_tensor_idx)
    
    #Create a tensor for the "YUV to RGB conv ":
    
    #conv filter/weight tensor data
    wights, bias = getWightsAndBiasData()
    conv_filter_tensor = createTensor(modelT, tflite_model.TensorType.TensorType.FLOAT32, None, [3,1,1,3], bytearray(str("YUV-RGB-Conv_weights"),'utf-8'))
    conv_filter_buffer = copy.copy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].buffer])
    conv_filter_buffer.data = struct.pack('%sf' % len(wights), *wights)
    modelT.buffers.append(conv_filter_buffer)
    conv_filter_buffer_idx = len(modelT.buffers) - 1
    conv_filter_tensor.buffer = conv_filter_buffer_idx
    modelT.subgraphs[0].tensors.append(conv_filter_tensor)
    conv_filter_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 

    #conv bias tensor data
    conv_bias_tensor = createTensor(modelT, tflite_model.TensorType.TensorType.FLOAT32, None, [3], bytearray(str("YUV-RGB-Conv_bias"),'utf-8'))
    conv_bias_buffer = copy.copy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].buffer])
    conv_bias_buffer.data = struct.pack('%sf' % len(bias), *bias)
    modelT.buffers.append(conv_bias_buffer)
    conv_bias_buffer_idx = len(modelT.buffers) - 1
    conv_bias_tensor.buffer = conv_bias_buffer_idx
    modelT.subgraphs[0].tensors.append(conv_bias_tensor)
    conv_bias_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 

    conv_tensor = createTensor(modelT, tflite_model.TensorType.TensorType.FLOAT32, None, shape_cp, bytearray(str("YUV-RGB-Conv"),'utf-8'))
    conv_buffer = copy.copy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].buffer])
    modelT.buffers.append(conv_buffer)
    conv_buffer_idx = len(modelT.buffers) - 1
    conv_tensor.buffer = conv_buffer_idx
    modelT.subgraphs[0].tensors.append(conv_tensor)
    conv_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 
    conv_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
    modelT.subgraphs[0].operators.insert(0,conv_op)
    
    modelT.subgraphs[0].operators[0].inputs = modelT.subgraphs[0].inputs
    modelT.subgraphs[0].operators[0].inputs = appendInput(modelT.subgraphs[0].operators[0].inputs,conv_bias_tensor_idx)
    modelT.subgraphs[0].operators[0].inputs = appendInput(modelT.subgraphs[0].operators[0].inputs,conv_filter_tensor_idx)    
    modelT.subgraphs[0].operators[0].outputs[0] =  conv_tensor_idx
    modelT.subgraphs[0].operators[1].inputs = removeInput(modelT.subgraphs[0].operators[1].inputs,modelT.subgraphs[0].inputs[0])
    modelT.subgraphs[0].operators[1].inputs = removeInput(modelT.subgraphs[0].operators[1].inputs,modelT.subgraphs[0].inputs[1])
    modelT.subgraphs[0].operators[1].inputs = appendInput(modelT.subgraphs[0].operators[1].inputs,conv_tensor_idx)
    conv_idx = addNewOperator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.CONV_2D)
    modelT.subgraphs[0].operators[0].opcodeIndex = conv_idx
    modelT.subgraphs[0].operators[0].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.Conv2DOptions
    modelT.subgraphs[0].operators[0].builtinOptions = tflite_model.Conv2DOptions.Conv2DOptionsT()
    modelT.subgraphs[0].operators[0].builtinOptions.strideH = 1
    modelT.subgraphs[0].operators[0].builtinOptions.strideW = 1

    #Create a tensor for the "YUVconcat ":
    concat_tensor = createTensor(modelT, tflite_model.TensorType.TensorType.FLOAT32, None, shape_cp, bytearray(str("YUV-concat"),'utf-8'))
    concat_buffer = copy.copy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].buffer])    
    modelT.buffers.append(concat_buffer)
    concat_buffer_idx = len(modelT.buffers) - 1
    concat_tensor.buffer = concat_buffer_idx
    modelT.subgraphs[0].tensors.append(concat_tensor)
    concat_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 
    concat_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
    modelT.subgraphs[0].operators.insert(0,concat_op)
    modelT.subgraphs[0].operators[0].inputs =  modelT.subgraphs[0].inputs
    modelT.subgraphs[0].operators[0].outputs[0] =  concat_tensor_idx
    # modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]].shapeSignature = [-1,224,224,3]
    modelT.subgraphs[0].operators[1].inputs = removeInput(modelT.subgraphs[0].operators[1].inputs,modelT.subgraphs[0].inputs[0])
    modelT.subgraphs[0].operators[1].inputs = removeInput(modelT.subgraphs[0].operators[1].inputs,modelT.subgraphs[0].inputs[1])
    modelT.subgraphs[0].operators[1].inputs = appendInput(modelT.subgraphs[0].operators[1].inputs,concat_tensor_idx)
    modelT.subgraphs[0].operators[1].inputs.reverse()
    
    concat_idx = addNewOperator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.CONCATENATION)
    modelT.subgraphs[0].operators[0].opcodeIndex = concat_idx
    modelT.subgraphs[0].operators[0].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.ConcatenationOptions
    modelT.subgraphs[0].operators[0].builtinOptions = tflite_model.ConcatenationOptions.ConcatenationOptionsT()
    modelT.subgraphs[0].operators[0].builtinOptions.axis = 3
    
    
    # Add resize Operator for upsampling:
    resize_tensor = createTensor(modelT, tflite_model.TensorType.TensorType.FLOAT32, None, [shape_cp[0], shape_cp[1], shape_cp[2], 2], bytearray(str("UV-Upsample"),'utf-8'))
    resize_buffer = copy.copy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].buffer])    
    modelT.buffers.append(resize_buffer)
    resize_buffer_idx = len(modelT.buffers) - 1
    resize_tensor.buffer = resize_buffer_idx
    modelT.subgraphs[0].tensors.append(resize_tensor)
    resize_tensor_idx = len(modelT.subgraphs[0].tensors) - 1 
    size_tensor = createTensor(modelT, tflite_model.TensorType.TensorType.INT32, None, [2], bytearray(str("Convert-YUV-RGB"),'utf-8'))
    size_buffer = copy.copy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].buffer])    
    data = [shape_cp[1],shape_cp[2]]
    size_buffer.data = struct.pack('%si' % len(data), *data)
    modelT.buffers.append(size_buffer)
    size_buffer_idx = len(modelT.buffers) - 1
    size_tensor.buffer = size_buffer_idx
    modelT.subgraphs[0].tensors.append(size_tensor)
    size_tensor_idx = len(modelT.subgraphs[0].tensors) - 1   

    resize_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
    modelT.subgraphs[0].operators.insert(0,resize_op)
    modelT.subgraphs[0].operators[0].inputs =  removeInput(modelT.subgraphs[0].inputs,modelT.subgraphs[0].inputs[0])    
    modelT.subgraphs[0].operators[0].outputs[0] =  resize_tensor_idx
    # modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]].shapeSignature = [-1,224,224,2]
    modelT.subgraphs[0].operators[1].inputs = removeInput(modelT.subgraphs[0].operators[1].inputs,modelT.subgraphs[0].inputs[1]) 
    modelT.subgraphs[0].operators[1].inputs = appendInput(modelT.subgraphs[0].operators[1].inputs,resize_tensor_idx)  
    modelT.subgraphs[0].operators[0].inputs = appendInput(modelT.subgraphs[0].operators[0].inputs,size_tensor_idx)    
    modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[1].outputs[0]].shape = shape_cp
    resize_idx = addNewOperator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR)
    modelT.subgraphs[0].operators[0].opcodeIndex = resize_idx
    modelT.subgraphs[0].operators[0].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.ResizeNearestNeighborOptions
    modelT.subgraphs[0].operators[0].builtinOptions = tflite_model.ResizeNearestNeighborOptions.ResizeNearestNeighborOptionsT()   
  
    #saving the model
    b2 = flatbuffers.Builder(0)
    b2.Finish(modelT.Pack(b2), b"TFL3")
    modelBuf = b2.Output() 
    newFile = open(out_model_path, "wb")
    newFile.write(modelBuf)
 

def main(argv):
   inputfile = ''
   outputfile = ''
   opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   if(inputfile == ''):
        print ('test.py -i <inputfile> -o <outputfile>')
        sys.exit()
   if(outputfile == ''):
        temp = inputfile
        outputfile = temp.replace('.tflite', '_yuv.tflite')
   print("inputfile: "+ inputfile)
   print("outputfile: "+ outputfile)
   addYUVConv(inputfile, outputfile)
if __name__ == "__main__":
   main(sys.argv[1:])